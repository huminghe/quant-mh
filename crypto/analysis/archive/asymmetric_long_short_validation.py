"""
非对称多空仓位比例验证（2026-06-03）

研究问题：
  加密货币存在结构性正漂移（长期价格上涨偏置），做多/做空等权分配是否次优？
  将做多仓位乘以 long_mult，做空仓位乘以 short_mult（整体平均资本占用不变），
  是否能改善 Sharpe 和总 PnL？

方案设计：
  - 约束：long_mult × P(long) + short_mult × P(short) ≈ 1（平均仓位保持在 BASE 附近）
  - 参数扫描：long_mult ∈ [0.8, 1.5]，short_mult 由约束推导，需要 P(long) 和 P(short)
  - 另外单独测试几个典型比例：50/50、60/40、70/30、80/20

注意：
  - 需要从 xlsx 读取方向（做多/做空），不能只用 pnl 正负（亏损的多单 pnl 为负）
  - TV 导出时间是 UTC+8 naive datetime，需要 -8h 转 UTC

基准对比：
  1. 固定仓位等权（BASE = 10000 USDT，多空相同）
  2. ATR 归一化定仓（2H ATR30 窗口500 cap=4x）+ 等权多空

数据：BTC/ETH/SOL/DOGE，v1+v2，2019-2026
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

BASE_CAPITAL = 10_000
downloads = Path('/Users/huminghe/Downloads')

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

# ─── 数据加载（带方向信息）────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，含方向（long/short），TV UTC+8 -8h 转 UTC"""
    path = downloads / fname
    if not path.exists():
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {name: i for i, name in enumerate(rows[0]) if name is not None}

    # 查找价格列（列名可能是"价格 USDT"等）
    price_key = next((k for k in col_idx if '价格' in str(k) or k == 'Price'), None)
    pnl_key = next((k for k in col_idx if '净损益' in str(k) and 'USDT' in str(k)), None) or \
              next((k for k in col_idx if 'Profit' in str(k) and 'USDT' in str(k)), None)

    # "信号"列：进场行里存放方向（'long'/'short'）
    signal_key = next((k for k in col_idx if '信号' in str(k) or k == 'Signal'), None)

    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx.get('交易 #', col_idx.get('Trade #', -1))]
            typ = str(row[col_idx.get('类型', col_idx.get('Type', -1))])
            dt  = row[col_idx.get('日期和时间', col_idx.get('Date/Time', -1))]
            pnl = row[col_idx.get(pnl_key, -1)] if pnl_key else None
            sig = str(row[col_idx[signal_key]]).lower() if signal_key else ''
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
                # 方向从"信号"列读取（long/short）
                if 'long' in sig:
                    by_num[num]['direction'] = 'long'
                elif 'short' in sig:
                    by_num[num]['direction'] = 'short'
                # 兜底：从"类型"列读取
                elif '多头' in typ or 'Long' in typ:
                    by_num[num]['direction'] = 'long'
                elif '空头' in typ or 'Short' in typ:
                    by_num[num]['direction'] = 'short'
                else:
                    by_num[num]['direction'] = 'unknown'
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except:
            continue
    wb.close()

    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    if not rows_out:
        return pd.DataFrame()
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    if 'direction' not in df.columns:
        df['direction'] = 'unknown'
    return df.sort_values('entry_dt').reset_index(drop=True)

# ─── OHLCV（ATR 基准用）──────────────────────────────────────────────────────

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf='2h'):
    key = (symbol, tf)
    if key in ohlcv_cache: return ohlcv_cache[key]
    print(f'  拉取 {symbol} {tf}...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                break
            except Exception as e:
                if attempt == 2: raise
                import time; time.sleep(3)
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

def alloc_atr_normalized(ohlcv_df, atr_n=30, median_window=500, cap_mult=4.0):
    atr = calc_atr(ohlcv_df, atr_n)
    atr_pct = atr / ohlcv_df['close']
    median_atr = atr_pct.rolling(median_window, min_periods=atr_n).median()
    raw = BASE_CAPITAL * (median_atr / atr_pct)
    return raw.clip(BASE_CAPITAL / cap_mult, BASE_CAPITAL * cap_mult)

# ─── 统计指标 ─────────────────────────────────────────────────────────────────

def stats(pnl_series, n_years=7, n_strats=8):
    if len(pnl_series) == 0:
        return dict(total=0, win_rate=0, rr=0, max_dd=0, sharpe=0)
    total = pnl_series.sum()
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    win_rate = len(wins) / len(pnl_series) * 100
    rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
    cum = pnl_series.cumsum()
    dd = ((cum - cum.cummax()) / (BASE_CAPITAL * n_strats) * 100).min()
    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series) / n_years) \
             if pnl_series.std() > 0 else 0
    return dict(total=total, win_rate=win_rate, rr=rr, max_dd=dd, sharpe=sharpe)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== 非对称多空仓位比例验证 ===\n")

    # 加载所有交易数据
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df
                n_long = (df['direction'] == 'long').sum()
                n_short = (df['direction'] == 'short').sum()
                n_unk = (df['direction'] == 'unknown').sum()
                print(f"  {ver} {sym}：{len(df)} 笔  做多={n_long}  做空={n_short}  未知={n_unk}")
    print(f"\n共 {len(all_trades)} 个策略×标的组合\n")

    # 方向分布统计（全局）
    all_directions = pd.concat([df['direction'] for df in all_trades.values()])
    n_long_total = (all_directions == 'long').sum()
    n_short_total = (all_directions == 'short').sum()
    n_total = len(all_directions)
    p_long = n_long_total / n_total
    p_short = n_short_total / n_total
    print(f"全局方向分布：做多 {n_long_total} 笔 ({p_long*100:.1f}%)，做空 {n_short_total} 笔 ({p_short*100:.1f}%)\n")

    # 拉取 OHLCV（ATR 基准用）
    print("拉取行情数据...")
    symbols = set(ccxt_sym for _, _, ccxt_sym in all_trades.keys())
    for sym in symbols:
        fetch_ohlcv(sym, '2h')
    print()

    # ── 基准1：固定仓位等权 ────────────────────────────────────────────────────
    fixed_pnl_all = pd.Series([p for df in all_trades.values() for p in df['pnl'].values])
    s_fixed = stats(fixed_pnl_all)
    print(f"基准1 固定仓位（{BASE_CAPITAL} USDT，多空等权）：")
    print(f"  总PnL={s_fixed['total']:+.0f}  胜率={s_fixed['win_rate']:.1f}%  "
          f"盈亏比={s_fixed['rr']:.2f}  最大回撤={s_fixed['max_dd']:.1f}%  Sharpe={s_fixed['sharpe']:.3f}\n")

    # ── 基准2：ATR 归一化等权 ──────────────────────────────────────────────────
    atr_pnl_all = []
    atr_alloc_map = {}
    for (ver, sym, ccxt_sym), trades_df in all_trades.items():
        ohlcv = ohlcv_cache[(ccxt_sym, '2h')]
        alloc = alloc_atr_normalized(ohlcv)
        alloc_df = alloc.rename('alloc').reset_index()
        alloc_df.columns = ['ts', 'alloc']
        alloc_df['ts'] = alloc_df['ts'].astype('datetime64[ns]')
        merged = pd.merge_asof(
            trades_df.sort_values('entry_dt'),
            alloc_df.sort_values('ts'),
            left_on='entry_dt', right_on='ts', direction='backward'
        )
        merged['pnl_atr'] = merged['pnl'] * (merged['alloc'] / BASE_CAPITAL)
        atr_pnl_all.extend(merged['pnl_atr'].values)
        atr_alloc_map[(ver, sym, ccxt_sym)] = merged[['entry_dt', 'pnl', 'direction', 'alloc']].copy()
    atr_pnl_all = pd.Series(atr_pnl_all)
    s_atr = stats(atr_pnl_all)
    print(f"基准2 ATR归一化等权（2H ATR30 窗口500 cap=4x）：")
    print(f"  总PnL={s_atr['total']:+.0f}  胜率={s_atr['win_rate']:.1f}%  "
          f"盈亏比={s_atr['rr']:.2f}  最大回撤={s_atr['max_dd']:.1f}%  Sharpe={s_atr['sharpe']:.3f}\n")

    # ── 逐方向拆解基准（了解多空各自贡献）─────────────────────────────────────
    all_merged = pd.concat([df for df in atr_alloc_map.values()], ignore_index=True)
    for direction in ['long', 'short']:
        sub = all_merged[all_merged['direction'] == direction]['pnl_atr'] if 'pnl_atr' in all_merged.columns else \
              all_merged[all_merged['direction'] == direction]['pnl']
        # 用 ATR pnl
        sub_pnl = all_merged[all_merged['direction'] == direction]['pnl_atr'] if 'pnl_atr' in all_merged.columns else pd.Series()
        # 需要手动计算 pnl_atr
    # 重新计算多空分拆
    long_pnl_atr = all_merged[all_merged['direction'] == 'long']['pnl_atr'] if 'pnl_atr' in all_merged.columns else pd.Series()
    short_pnl_atr = all_merged[all_merged['direction'] == 'short']['pnl_atr'] if 'pnl_atr' in all_merged.columns else pd.Series()

    # all_merged 里有 pnl 和 alloc，补算 pnl_atr
    all_merged['pnl_atr'] = all_merged['pnl'] * (all_merged['alloc'] / BASE_CAPITAL)
    long_pnl_atr  = all_merged[all_merged['direction'] == 'long']['pnl_atr']
    short_pnl_atr = all_merged[all_merged['direction'] == 'short']['pnl_atr']

    s_long  = stats(long_pnl_atr,  n_years=7, n_strats=4)
    s_short = stats(short_pnl_atr, n_years=7, n_strats=4)
    print("── 多空分拆（ATR定仓基准）──")
    print(f"  做多：总PnL={s_long['total']:+.0f}  胜率={s_long['win_rate']:.1f}%  "
          f"盈亏比={s_long['rr']:.2f}  Sharpe={s_long['sharpe']:.3f}")
    print(f"  做空：总PnL={s_short['total']:+.0f}  胜率={s_short['win_rate']:.1f}%  "
          f"盈亏比={s_short['rr']:.2f}  Sharpe={s_short['sharpe']:.3f}\n")

    # ── 非对称多空参数扫描 ─────────────────────────────────────────────────────
    # 约束：期望平均仓位不变 → long_mult × p_long + short_mult × p_short = 1
    # 给定 long_mult，推导 short_mult = (1 - long_mult × p_long) / p_short
    print("─── 非对称多空参数扫描（ATR定仓 × 方向乘数）───")
    print(f"{'多头乘数':>8s}  {'空头乘数':>8s}  {'总PnL':>10s}  {'最大回撤':>8s}  "
          f"{'Sharpe':>8s}  {'vs ATR':>8s}  {'vs固定':>8s}")
    print("─" * 75)

    results = []
    # 扫描 long_mult 从 0.7 到 1.5
    for long_mult in np.arange(0.7, 1.55, 0.05):
        short_mult = (1.0 - long_mult * p_long) / p_short if p_short > 0 else 1.0
        if short_mult < 0.1 or short_mult > 3.0:
            continue  # 跳过不合理的参数

        pnl_scaled = all_merged['pnl_atr'] * np.where(
            all_merged['direction'] == 'long', long_mult,
            np.where(all_merged['direction'] == 'short', short_mult, 1.0)
        )
        s = stats(pd.Series(pnl_scaled), n_years=7, n_strats=8)
        vs_atr   = s['sharpe'] - s_atr['sharpe']
        vs_fixed = s['sharpe'] - s_fixed['sharpe']
        results.append(dict(long_mult=long_mult, short_mult=short_mult, **s,
                            vs_atr=vs_atr, vs_fixed=vs_fixed))
        marker = ' ◀ 最优' if len(results) == 1 else ''
        print(f"{long_mult:>8.2f}x  {short_mult:>8.2f}x  {s['total']:>+10.0f}  "
              f"{s['max_dd']:>7.1f}%  {s['sharpe']:>8.3f}  {vs_atr:>+8.3f}  {vs_fixed:>+8.3f}")

    # ── 找最优 ─────────────────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda r: r['sharpe'])
        print(f"\n最优：long_mult={best['long_mult']:.2f}x  short_mult={best['short_mult']:.2f}x  "
              f"Sharpe={best['sharpe']:.3f}  vs ATR={best['vs_atr']:+.3f}")

    # ── 几个典型比例的对比（固定倍数，不受多空比例约束）─────────────────────────
    print("\n─── 典型比例对比（固定多空乘数，不归一化均值仓位）───")
    print(f"{'方案':>12s}  {'多乘数':>6s}  {'空乘数':>6s}  {'总PnL':>10s}  "
          f"{'最大回撤':>8s}  {'Sharpe':>8s}  {'vs ATR':>8s}")
    print("─" * 70)
    for name, lm, sm in [
        ('50/50等权', 1.0, 1.0),
        ('60/40', 1.2, 0.8),
        ('70/30', 1.4, 0.6),
        ('80/20', 1.6, 0.4),
        ('仅做多', 2.0, 0.0),
        ('空头×0.5', 1.0, 0.5),
    ]:
        pnl_scaled = all_merged['pnl_atr'] * np.where(
            all_merged['direction'] == 'long', lm,
            np.where(all_merged['direction'] == 'short', sm, 1.0)
        )
        s = stats(pd.Series(pnl_scaled), n_years=7, n_strats=8)
        vs_atr = s['sharpe'] - s_atr['sharpe']
        print(f"{name:>12s}  {lm:>6.1f}x  {sm:>6.1f}x  {s['total']:>+10.0f}  "
              f"{s['max_dd']:>7.1f}%  {s['sharpe']:>8.3f}  {vs_atr:>+8.3f}")

    # ── 按时间段分析（牛市/熊市分别看）────────────────────────────────────────
    print("\n─── 分时段分析（70/30 vs 等权，ATR定仓基础上）───")
    periods = [
        ('2019-2020（震荡期）', '2019-01-01', '2020-10-01'),
        ('2020-2021（超级牛市）', '2020-10-01', '2021-12-31'),
        ('2022（熊市）', '2022-01-01', '2022-12-31'),
        ('2023-2024（复苏牛市）', '2023-01-01', '2024-12-31'),
        ('2025-2026（震荡期）', '2025-01-01', '2026-06-03'),
    ]
    print(f"{'时段':>22s}  {'等权Sharpe':>10s}  {'70/30 Sharpe':>12s}  {'差值':>8s}")
    print("─" * 60)
    for period_name, start, end in periods:
        mask = (all_merged['entry_dt'] >= start) & (all_merged['entry_dt'] < end)
        sub = all_merged[mask]
        if len(sub) < 10:
            continue
        # 等权
        s_eq = stats(sub['pnl_atr'], n_years=(pd.Timestamp(end)-pd.Timestamp(start)).days/365, n_strats=8)
        # 70/30（lm=1.4, sm=0.6）
        pnl_7030 = sub['pnl_atr'] * np.where(
            sub['direction'] == 'long', 1.4,
            np.where(sub['direction'] == 'short', 0.6, 1.0)
        )
        s_7030 = stats(pd.Series(pnl_7030), n_years=(pd.Timestamp(end)-pd.Timestamp(start)).days/365, n_strats=8)
        diff = s_7030['sharpe'] - s_eq['sharpe']
        print(f"{period_name:>22s}  {s_eq['sharpe']:>10.3f}  {s_7030['sharpe']:>12.3f}  {diff:>+8.3f}")

if __name__ == '__main__':
    run()
