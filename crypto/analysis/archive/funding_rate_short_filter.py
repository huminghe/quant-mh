"""
资金费率空头方向过滤验证（2026-06-03）

与上次验证（funding_rate_filter.py，结论 -6.4%）的区别：
  上次：泛化过滤，双向都加费率条件，逻辑是"避免拥挤交易"
  本次：仅针对空头方向，逻辑是"资金费率为正时做空收款，费率越高空头越有利"

核心问题：
  1. 入场时资金费率水平是否能预测空头交易质量？
     （描述性分析：按 FR 分桶，看空头平均 PnL）
  2. "FR > 阈值才允许做空"是否改善 Sharpe？
     多头不受限制（FR 对多头影响方向相反，但上次已验证多头过滤无效）

基准：
  ATR 归一化定仓（2H ATR30，窗口500，cap=4x），Sharpe 3.178

数据：BTC/ETH/SOL/DOGE，v1+v2，2019-2026
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl, time
from pathlib import Path

BASE_CAPITAL = 10_000
downloads    = Path('/Users/huminghe/Downloads')

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
def fetch_ohlcv(symbol, tf='2h'):
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

fr_cache = {}
def fetch_funding_rate(symbol):
    if symbol in fr_cache: return fr_cache[symbol]
    print(f'  拉取 {symbol} 资金费率...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    all_rates, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        rates = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not rates: break
        all_rates.extend(rates)
        if len(rates) < 1000: break
        since = rates[-1]['timestamp'] + 1
        time.sleep(0.05)
    df = pd.DataFrame([{
        'dt': pd.Timestamp(r['datetime']).tz_localize(None),
        'fr': r['fundingRate']
    } for r in all_rates])
    df = df.set_index('dt').sort_index()
    fr_cache[symbol] = df
    print(f'done ({len(df)} 条)')
    return df

def calc_atr(df, n=30):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def alloc_atr(ohlcv_df, atr_n=30, median_window=500, cap=4.0):
    atr_pct = calc_atr(ohlcv_df, atr_n) / ohlcv_df['close']
    med     = atr_pct.rolling(median_window, min_periods=atr_n).median()
    raw     = BASE_CAPITAL * (med / atr_pct)
    return raw.clip(BASE_CAPITAL / cap, BASE_CAPITAL * cap)

def merge_val(trades, ref_series, col='val'):
    """取入场时刻前一个已知值"""
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
    print("=== 资金费率空头方向过滤验证 ===\n")

    # 加载数据
    print("加载交易记录...")
    all_rows = []
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if df.empty: continue
            df['ver'] = ver; df['sym'] = sym; df['ccxt_sym'] = ccxt_sym
            all_rows.append(df)
            print(f"  {ver} {sym}: {len(df)} 笔  多={( df['direction']=='long').sum()}  空={(df['direction']=='short').sum()}")

    print("\n拉取行情与资金费率数据...")
    symbols = ['BTC/USDT','ETH/USDT','SOL/USDT','DOGE/USDT']
    for sym in symbols:
        fetch_ohlcv(sym, '2h')
        fetch_funding_rate(sym)

    # 给每笔交易附上 ATR 定仓和资金费率
    enriched = []
    for df in all_rows:
        ccxt_sym = df['ccxt_sym'].iloc[0]
        ohlcv = ohlcv_cache[(ccxt_sym, '2h')]
        fr_df = fr_cache[ccxt_sym]

        atr_alloc = alloc_atr(ohlcv)
        df['alloc'] = merge_val(df, atr_alloc).values
        df['pnl_atr'] = df['pnl'] * (df['alloc'] / BASE_CAPITAL)

        # 资金费率：原始 + 3期均值（24H）+ 7期均值（56H）
        df['fr']     = merge_val(df, fr_df['fr']).values
        df['fr_ma3'] = merge_val(df, fr_df['fr'].rolling(3).mean()).values
        df['fr_ma7'] = merge_val(df, fr_df['fr'].rolling(7).mean()).values

        enriched.append(df)

    all_df = pd.concat(enriched, ignore_index=True)
    short_df = all_df[all_df['direction'] == 'short'].copy()
    long_df  = all_df[all_df['direction'] == 'long'].copy()
    print(f"\n共 {len(all_df)} 笔  多头 {len(long_df)} 笔  空头 {len(short_df)} 笔\n")

    # ── 基准 ──────────────────────────────────────────────────────────────────
    s_base = stats(all_df['pnl_atr'])
    s_long = stats(long_df['pnl_atr'], n_strats=4)
    s_short = stats(short_df['pnl_atr'], n_strats=4)
    print("── ATR定仓基准 ──")
    print(f"  全部：总PnL={s_base['total']:+.0f}  胜率={s_base['win_rate']:.1f}%  "
          f"盈亏比={s_base['rr']:.2f}  最大回撤={s_base['max_dd']:.1f}%  Sharpe={s_base['sharpe']:.3f}")
    print(f"  多头：总PnL={s_long['total']:+.0f}  胜率={s_long['win_rate']:.1f}%  "
          f"盈亏比={s_long['rr']:.2f}  Sharpe={s_long['sharpe']:.3f}")
    print(f"  空头：总PnL={s_short['total']:+.0f}  胜率={s_short['win_rate']:.1f}%  "
          f"盈亏比={s_short['rr']:.2f}  Sharpe={s_short['sharpe']:.3f}\n")

    # ── 资金费率分布（空头入场时刻）─────────────────────────────────────────
    fr_short = short_df['fr'].dropna()
    print("── 空头入场时刻资金费率分布 ──")
    print(f"  中位数: {fr_short.median()*100:.4f}%  均值: {fr_short.mean()*100:.4f}%")
    print(f"  >0 占比: {(fr_short>0).mean()*100:.1f}%  <0 占比: {(fr_short<0).mean()*100:.1f}%")
    print(f"  >0.01% 占比: {(fr_short>0.0001).mean()*100:.1f}%")
    print(f"  >0.03% 占比: {(fr_short>0.0003).mean()*100:.1f}%")
    print(f"  <0 占比（付费做空）: {(fr_short<0).mean()*100:.1f}%\n")

    # ── 描述性分析：按FR分桶看空头PnL ─────────────────────────────────────
    print("── 空头交易按入场时刻资金费率分桶（ATR定仓）──")
    bins = [-np.inf, -0.0003, -0.0001, 0, 0.0001, 0.0003, 0.001, np.inf]
    labels = ['<-0.03%', '-0.03~-0.01%', '-0.01~0%', '0~0.01%', '0.01~0.03%', '0.03~0.1%', '>0.1%']
    short_valid = short_df.dropna(subset=['fr']).copy()
    short_valid['fr_bucket'] = pd.cut(short_valid['fr'], bins=bins, labels=labels)
    bucket_stats = short_valid.groupby('fr_bucket', observed=True)['pnl_atr'].agg(
        count='count',
        total='sum',
        mean='mean',
        win_rate=lambda x: (x > 0).mean() * 100
    ).reset_index()
    print(f"  {'费率区间':>16s}  {'笔数':>5s}  {'均值PnL':>8s}  {'胜率':>6s}  {'总PnL':>9s}")
    print("  " + "─" * 52)
    for _, row in bucket_stats.iterrows():
        print(f"  {str(row['fr_bucket']):>16s}  {row['count']:>5.0f}  {row['mean']:>8.1f}  "
              f"{row['win_rate']:>5.1f}%  {row['total']:>+9.1f}")

    # 同样做多头的分桶（对照）
    print("\n── 多头交易按入场时刻资金费率分桶（ATR定仓，对照）──")
    long_valid = long_df.dropna(subset=['fr']).copy()
    long_valid['fr_bucket'] = pd.cut(long_valid['fr'], bins=bins, labels=labels)
    bucket_long = long_valid.groupby('fr_bucket', observed=True)['pnl_atr'].agg(
        count='count', mean='mean', win_rate=lambda x: (x > 0).mean() * 100
    ).reset_index()
    print(f"  {'费率区间':>16s}  {'笔数':>5s}  {'均值PnL':>8s}  {'胜率':>6s}")
    print("  " + "─" * 42)
    for _, row in bucket_long.iterrows():
        print(f"  {str(row['fr_bucket']):>16s}  {row['count']:>5.0f}  {row['mean']:>8.1f}  "
              f"{row['win_rate']:>5.1f}%")

    # ── 过滤效果：仅限空头，多头不受限 ──────────────────────────────────────
    print("\n── 空头过滤参数扫描（多头不受影响，Sharpe基准=3.178）──")
    print(f"  {'过滤条件':>30s}  {'空头保留率':>9s}  {'总Sharpe':>9s}  {'vs基准':>8s}  {'空头Sharpe':>10s}")
    print("  " + "─" * 72)

    best_sharpe, best_label = -np.inf, ''
    results = []

    for fr_col, fr_label in [('fr', '原始FR'), ('fr_ma3', 'FR_MA3'), ('fr_ma7', 'FR_MA7')]:
        for threshold in [-0.0003, -0.0002, -0.0001, 0.0, 0.0001, 0.0002, 0.0003, 0.0005]:
            # 过滤逻辑：空头交易要求 FR > threshold
            short_mask = (short_df[fr_col].notna()) & (short_df[fr_col] > threshold)
            keep_rate  = short_mask.mean()
            if keep_rate < 0.05:
                continue  # 保留率过低，跳过

            # 重组：多头全保留 + 空头过滤后
            filtered_pnl = pd.concat([
                long_df['pnl_atr'],
                short_df.loc[short_mask, 'pnl_atr']
            ])
            s_all   = stats(filtered_pnl)
            s_short_f = stats(short_df.loc[short_mask, 'pnl_atr'], n_strats=4)
            vs_base = s_all['sharpe'] - s_base['sharpe']

            label = f"{fr_label} > {threshold*100:.3f}%"
            results.append(dict(label=label, keep_rate=keep_rate, sharpe=s_all['sharpe'],
                                vs_base=vs_base, short_sharpe=s_short_f['sharpe']))
            if s_all['sharpe'] > best_sharpe:
                best_sharpe, best_label = s_all['sharpe'], label

            print(f"  {label:>30s}  {keep_rate*100:>8.1f}%  {s_all['sharpe']:>9.3f}  "
                  f"{vs_base:>+8.3f}  {s_short_f['sharpe']:>10.3f}")
        print()

    # ── 最优结果 ──────────────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda r: r['sharpe'])
        print(f"最优：{best['label']}  空头保留率={best['keep_rate']*100:.1f}%  "
              f"Sharpe={best['sharpe']:.3f}  vs基准={best['vs_base']:+.3f}")

    # ── 分时段分析（看哪个市场环境下有效）──────────────────────────────────
    print("\n── 分时段分析（FR>0 空头过滤 vs 无过滤，ATR定仓）──")
    periods = [
        ('2019-2020（震荡）', '2019-01-01', '2020-10-01'),
        ('2020-2021（牛市）', '2020-10-01', '2021-12-31'),
        ('2022（熊市）',      '2022-01-01', '2022-12-31'),
        ('2023-2024（牛市）', '2023-01-01', '2024-12-31'),
        ('2025-2026（震荡）', '2025-01-01', '2026-06-03'),
    ]
    print(f"  {'时段':>18s}  {'无过滤Sharpe':>12s}  {'FR>0过滤Sharpe':>14s}  {'差值':>6s}  {'空头保留':>8s}")
    print("  " + "─" * 68)
    for period_name, start, end in periods:
        mask_period = (all_df['entry_dt'] >= start) & (all_df['entry_dt'] < end)
        sub = all_df[mask_period]
        if len(sub) < 20: continue

        sub_short = sub[sub['direction'] == 'short']
        sub_long  = sub[sub['direction'] == 'long']
        n_years   = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365

        s_eq = stats(sub['pnl_atr'], n_years=n_years)

        fr_mask = sub_short['fr'].notna() & (sub_short['fr'] > 0)
        keep    = fr_mask.mean()
        filtered = pd.concat([sub_long['pnl_atr'], sub_short.loc[fr_mask, 'pnl_atr']])
        s_f = stats(filtered, n_years=n_years)

        diff = s_f['sharpe'] - s_eq['sharpe']
        print(f"  {period_name:>18s}  {s_eq['sharpe']:>12.3f}  {s_f['sharpe']:>14.3f}  "
              f"{diff:>+6.3f}  {keep*100:>7.1f}%")

if __name__ == '__main__':
    run()
