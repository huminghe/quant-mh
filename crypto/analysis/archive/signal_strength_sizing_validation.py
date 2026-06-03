"""
信号强度加权仓位验证（2026-06-03）

研究问题：
  EMA 差值大小（|diff| / ATR）是否能预测单笔交易质量？
  用差值强度作为 ATR 定仓的额外乘数，强信号满仓、弱信号缩仓。

与入场过滤器的区别：
  过滤器是"拦截弱信号"（二元），缩仓是"弱信号给少钱"（连续），
  弱信号即使被过滤也会产生机会成本，缩仓保留所有信号只是减少暴露。

实现方式：
  scale = clip( |ema_diff_pct| / k, min_scale, max_scale )
  pnl_scaled = pnl_atr * scale
  其中 ema_diff_pct = |EMA_diff| / close（无量纲，类似 ATR%）

参数扫描：
  k（归一化系数）：0.5x~3.0x median(|diff_pct|) 分档
  min_scale（弱信号最低仓位）：0.2, 0.3, 0.5
  max_scale（强信号最高仓位）：1.5, 2.0（整体均值保持接近1）

基准：
  ATR 归一化定仓（2H ATR30，窗口500，cap=4x），Sharpe 3.178

数据：BTC/ETH/SOL/DOGE，v1+v2，2019-2026

注意：
  需要从 Binance 拉取入场时刻的 EMA 快慢线差值，
  TV 导出的 xlsx 没有存快慢线的具体数值，需要重建。
  EMA 参数：v1 策略 EMA(7)/EMA(25)，v2 策略参数略不同，
  但这里用统一的 EMA(7)/EMA(25) 测试方向性结论。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl, time
from pathlib import Path

BASE_CAPITAL = 10_000
downloads    = Path('/Users/huminghe/Downloads')

# v1: EMA(7)/EMA(25)，v2: 先用同样参数探索方向性
EMA_FAST = 7
EMA_SLOW = 25

VERSION_FILES = {
    'v1': {
        'BTC':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
        'ETH':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
        'SOL':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
        'DOGE': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
    },
    'v2': {
        'BTC':  ('v2_strategy_btc_OKX_BTCUSDT.P_2026-05-22_c2fde.xlsx',   'BTC/USDT'),
        'ETH':  ('v2_strategy_eth_OKX_ETHUSDT.P_2026-05-22_0f7c1.xlsx',   'ETH/USDT'),
        'SOL':  ('v2_strategy_sol_OKX_SOLUSDT.P_2026-05-22_6d6ed.xlsx',   'SOL/USDT'),
        'DOGE': ('v2_strategy_doge_OKX_DOGEUSDT.P_2026-05-22_8856b.xlsx', 'DOGE/USDT'),
    },
}

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，含方向，TV UTC+8 -8h 转 UTC"""
    path = downloads / fname
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {name: i for i, name in enumerate(rows[0]) if name is not None}
    signal_key = next((k for k in col_idx if '信号' in str(k) or k == 'Signal'), None)
    pnl_key    = next((k for k in col_idx if '净损益' in str(k) and 'USDT' in str(k)), None)
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx.get('交易 #', col_idx.get('Trade #', -1))]
            typ = str(row[col_idx.get('类型', col_idx.get('Type', -1))])
            dt  = row[col_idx.get('日期和时间', col_idx.get('Date/Time', -1))]
            pnl = row[col_idx[pnl_key]] if pnl_key else None
            sig = str(row[col_idx[signal_key]]).lower() if signal_key else ''
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
                by_num[num]['direction'] = 'long' if 'long' in sig else 'short'
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except:
            continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf='8h'):
    key = (symbol, tf)
    if key in ohlcv_cache: return ohlcv_cache[key]
    print(f'  拉取 {symbol} {tf}...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000); break
            except:
                if attempt == 2: raise
                time.sleep(3)
        if not bars: break
        all_bars.extend(bars)
        if len(bars) < 1000: break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[ns]')
    df = df.set_index('dt').sort_index()
    ohlcv_cache[key] = df
    print('done')
    return df

def calc_atr(df, n=30):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def alloc_atr(ohlcv_df, tf='2h', atr_n=30, median_window=500, cap=4.0):
    """ATR 归一化定仓（从2h数据计算）"""
    ohlcv_2h = ohlcv_cache.get((ohlcv_df.name if hasattr(ohlcv_df, 'name') else 'tmp', '2h'), ohlcv_df)
    atr_pct = calc_atr(ohlcv_2h, atr_n) / ohlcv_2h['close']
    med     = atr_pct.rolling(median_window, min_periods=atr_n).median()
    raw     = BASE_CAPITAL * (med / atr_pct)
    return raw.clip(BASE_CAPITAL / cap, BASE_CAPITAL * cap)

def merge_val(trades, ref_series, col='val'):
    """取入场时刻前一个已知值（merge_asof 向后）"""
    ref_df = ref_series.rename(col).reset_index()
    ref_df.columns = ['dt', col]
    ref_df['dt'] = ref_df['dt'].astype('datetime64[ns]')
    merged = pd.merge_asof(
        trades[['entry_dt']].sort_values('entry_dt'),
        ref_df.sort_values('dt'),
        left_on='entry_dt', right_on='dt', direction='backward'
    )
    return merged.set_index(trades.sort_values('entry_dt').index)[col].reindex(trades.index)

def stats(pnl_s, n_years=7, n_strats=8):
    if len(pnl_s) == 0:
        return dict(total=0, win_rate=0, rr=0, max_dd=0, sharpe=0, n=0)
    wins   = pnl_s[pnl_s > 0]
    losses = pnl_s[pnl_s < 0]
    cum    = pnl_s.cumsum()
    dd     = ((cum - cum.cummax()) / (BASE_CAPITAL * n_strats) * 100).min()
    sharpe = (pnl_s.mean() / pnl_s.std()) * np.sqrt(len(pnl_s) / n_years) if pnl_s.std() > 0 else 0
    return dict(
        total    = pnl_s.sum(),
        win_rate = len(wins) / len(pnl_s) * 100,
        rr       = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0,
        max_dd   = dd,
        sharpe   = sharpe,
        n        = len(pnl_s),
    )

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print(f"=== 信号强度加权仓位验证 ===")
    print(f"EMA 参数：fast={EMA_FAST}, slow={EMA_SLOW}（8H K线计算）\n")

    # 加载交易记录
    print("加载交易记录...")
    all_rows = []
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if df.empty: continue
            df['ver'] = ver; df['sym'] = sym; df['ccxt_sym'] = ccxt_sym
            all_rows.append(df)
            print(f"  {ver} {sym}: {len(df)} 笔")

    print("\n拉取行情数据（8H用于EMA，2H用于ATR定仓）...")
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    for sym in symbols:
        fetch_ohlcv(sym, '8h')
        fetch_ohlcv(sym, '2h')

    # 附上 ATR 定仓和 EMA 差值强度
    print("\n计算信号强度...")
    enriched = []
    for df in all_rows:
        ccxt_sym  = df['ccxt_sym'].iloc[0]
        ohlcv_8h  = ohlcv_cache[(ccxt_sym, '8h')]
        ohlcv_2h  = ohlcv_cache[(ccxt_sym, '2h')]

        # ATR 归一化定仓（2H）
        atr_pct  = calc_atr(ohlcv_2h, 30) / ohlcv_2h['close']
        med_atr  = atr_pct.rolling(500, min_periods=30).median()
        atr_alloc = (BASE_CAPITAL * (med_atr / atr_pct)).clip(BASE_CAPITAL / 4, BASE_CAPITAL * 4)

        # EMA 差值（8H）
        ema_fast  = ohlcv_8h['close'].ewm(span=EMA_FAST, adjust=False).mean()
        ema_slow  = ohlcv_8h['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        ema_diff  = ema_fast - ema_slow
        # 差值百分比（相对收盘价），衡量信号强度
        ema_diff_pct = ema_diff.abs() / ohlcv_8h['close']
        # ATR 百分比（8H，归一化用）
        atr_8h_pct   = calc_atr(ohlcv_8h, 14) / ohlcv_8h['close']

        df = df.copy()
        df['alloc']        = merge_val(df, atr_alloc).values
        df['pnl_atr']      = df['pnl'] * (df['alloc'] / BASE_CAPITAL)
        df['diff_pct']     = merge_val(df, ema_diff_pct).values      # |EMA diff| / close
        df['atr_8h_pct']   = merge_val(df, atr_8h_pct).values        # ATR% 8H，用于归一化 diff
        df['diff_over_atr']= df['diff_pct'] / df['atr_8h_pct']       # 无量纲信号强度

        enriched.append(df)

    all_df = pd.concat(enriched, ignore_index=True).dropna(subset=['diff_pct', 'atr_8h_pct'])
    print(f"有效样本：{len(all_df)} 笔（原 {sum(len(r) for r in all_rows)} 笔）\n")

    # ── 基准 ──────────────────────────────────────────────────────────────────
    s_base = stats(all_df['pnl_atr'])
    print(f"ATR定仓基准：总PnL={s_base['total']:+.0f}  胜率={s_base['win_rate']:.1f}%  "
          f"盈亏比={s_base['rr']:.2f}  最大回撤={s_base['max_dd']:.1f}%  Sharpe={s_base['sharpe']:.3f}\n")

    # ── 描述性：信号强度分桶看 PnL ────────────────────────────────────────────
    print("── 信号强度分桶（diff/ATR 分位数，看是否有单调关系）──")
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_q  = ['Q1(最弱)', 'Q2', 'Q3', 'Q4', 'Q5(最强)']
    all_df['strength_q'] = pd.qcut(all_df['diff_over_atr'], q=5, labels=labels_q)
    grp = all_df.groupby('strength_q', observed=True)['pnl_atr'].agg(
        count='count', mean='mean', win_rate=lambda x: (x > 0).mean() * 100, total='sum'
    ).reset_index()
    print(f"  {'分位':>10s}  {'笔数':>5s}  {'diff/ATR均值':>12s}  {'均值PnL':>8s}  {'胜率':>6s}  {'总PnL':>9s}")
    print("  " + "─" * 58)
    quantile_means = all_df.groupby('strength_q', observed=True)['diff_over_atr'].mean()
    for _, row in grp.iterrows():
        q_mean = quantile_means.get(row['strength_q'], np.nan)
        print(f"  {str(row['strength_q']):>10s}  {row['count']:>5.0f}  {q_mean:>12.3f}  "
              f"{row['mean']:>8.1f}  {row['win_rate']:>5.1f}%  {row['total']:>+9.1f}")

    # ── 描述性：diff_pct 绝对值分桶（看绝对值而非相对ATR）─────────────────
    print("\n── 原始 diff_pct 分桶（|EMA_diff|/close 分位数）──")
    all_df['diff_q'] = pd.qcut(all_df['diff_pct'], q=5, labels=labels_q)
    grp2 = all_df.groupby('diff_q', observed=True)['pnl_atr'].agg(
        count='count', mean='mean', win_rate=lambda x: (x > 0).mean() * 100, total='sum'
    ).reset_index()
    diff_means = all_df.groupby('diff_q', observed=True)['diff_pct'].mean()
    print(f"  {'分位':>10s}  {'笔数':>5s}  {'diff_pct均值':>12s}  {'均值PnL':>8s}  {'胜率':>6s}  {'总PnL':>9s}")
    print("  " + "─" * 58)
    for _, row in grp2.iterrows():
        d_mean = diff_means.get(row['diff_q'], np.nan)
        print(f"  {str(row['diff_q']):>10s}  {row['count']:>5.0f}  {d_mean:>12.4f}  "
              f"{row['mean']:>8.1f}  {row['win_rate']:>5.1f}%  {row['total']:>+9.1f}")

    # ── 参数扫描：信号强度乘数 ─────────────────────────────────────────────
    # scale = clip(diff_over_atr / k, min_s, max_s)，使 mean(scale) ≈ 1
    print("\n── 信号强度乘数参数扫描（pnl_atr × scale，Sharpe基准=3.178）──")
    print(f"  {'k':>5s}  {'min_s':>6s}  {'max_s':>6s}  {'均值scale':>9s}  "
          f"{'总PnL':>9s}  {'最大回撤':>8s}  {'Sharpe':>8s}  {'vs基准':>8s}")
    print("  " + "─" * 70)

    results = []
    median_strength = all_df['diff_over_atr'].median()
    print(f"  diff/ATR 中位数: {median_strength:.3f}，均值: {all_df['diff_over_atr'].mean():.3f}")
    print()

    for k_mult in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        k = median_strength * k_mult  # 以中位数的倍数为参考点
        for min_s in [0.2, 0.3, 0.5]:
            for max_s in [1.5, 2.0, 3.0]:
                if min_s >= max_s: continue
                scale = (all_df['diff_over_atr'] / k).clip(min_s, max_s)
                # 归一化：使 mean(scale) = 1，保持平均仓位不变
                scale = scale / scale.mean()
                # 重新 clip 防止归一化后超界
                scale = scale.clip(min_s / scale.mean() if scale.mean() > 0 else min_s,
                                   max_s / scale.mean() if scale.mean() > 0 else max_s)
                pnl_scaled = all_df['pnl_atr'] * scale
                s = stats(pnl_scaled)
                vs_base = s['sharpe'] - s_base['sharpe']
                results.append(dict(k_mult=k_mult, min_s=min_s, max_s=max_s,
                                    mean_scale=scale.mean(), sharpe=s['sharpe'], vs_base=vs_base))
                print(f"  {k_mult:>5.2f}x  {min_s:>6.1f}  {max_s:>6.1f}  {scale.mean():>9.3f}  "
                      f"{s['total']:>+9.0f}  {s['max_dd']:>7.1f}%  {s['sharpe']:>8.3f}  {vs_base:>+8.3f}")

    if results:
        best = max(results, key=lambda r: r['sharpe'])
        print(f"\n最优：k={best['k_mult']:.2f}x中位数  min={best['min_s']}  max={best['max_s']}  "
              f"Sharpe={best['sharpe']:.3f}  vs基准={best['vs_base']:+.3f}")

    # ── 分时段分析（最优参数 vs 无加权）──────────────────────────────────
    print("\n── 分时段分析（最优强度加权 vs 无加权，ATR定仓基础上）──")
    if results:
        best = max(results, key=lambda r: r['sharpe'])
        bk = median_strength * best['k_mult']
        bmin, bmax = best['min_s'], best['max_s']

        periods = [
            ('2019-2020（震荡）', '2019-01-01', '2020-10-01'),
            ('2020-2021（牛市）', '2020-10-01', '2021-12-31'),
            ('2022（熊市）',      '2022-01-01', '2022-12-31'),
            ('2023-2024（牛市）', '2023-01-01', '2024-12-31'),
            ('2025-2026（震荡）', '2025-01-01', '2026-06-03'),
        ]
        print(f"  {'时段':>18s}  {'无加权Sharpe':>12s}  {'加权Sharpe':>12s}  {'差值':>6s}")
        print("  " + "─" * 56)
        for period_name, start, end in periods:
            mask = (all_df['entry_dt'] >= start) & (all_df['entry_dt'] < end)
            sub  = all_df[mask]
            if len(sub) < 20: continue
            n_yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365
            s_eq  = stats(sub['pnl_atr'], n_years=n_yrs)
            scale = (sub['diff_over_atr'] / bk).clip(bmin, bmax)
            scale = scale / scale.mean()
            pnl_w = sub['pnl_atr'] * scale
            s_w   = stats(pnl_w, n_years=n_yrs)
            print(f"  {period_name:>18s}  {s_eq['sharpe']:>12.3f}  {s_w['sharpe']:>12.3f}  "
                  f"{s_w['sharpe']-s_eq['sharpe']:>+6.3f}")

    # ── 相关性检验：信号强度与 PnL 是否有正相关 ──────────────────────────
    print("\n── 信号强度与 PnL 相关性 ──")
    corr_raw  = all_df['diff_pct'].corr(all_df['pnl_atr'])
    corr_norm = all_df['diff_over_atr'].corr(all_df['pnl_atr'])
    corr_rank = all_df['diff_over_atr'].rank().corr(all_df['pnl_atr'].rank())  # 秩相关
    print(f"  Pearson(diff_pct,  pnl): {corr_raw:.4f}")
    print(f"  Pearson(diff/ATR,  pnl): {corr_norm:.4f}")
    print(f"  Spearman(diff/ATR, pnl): {corr_rank:.4f}")
    print(f"  （接近0表示无线性关系）")

if __name__ == '__main__':
    run()
