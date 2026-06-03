"""
资金费率空头过滤 IS/OOS 验证 + 各标的拆分（2026-06-03）

在 funding_rate_short_filter.py 结论基础上：
  全量最优：FR_MA7 > 0，Sharpe 3.178 → 3.359（+0.181）

本脚本额外验证：
  1. IS/OOS 分割（前80% IS，后20% OOS）
  2. 各标的（BTC/ETH/SOL/DOGE）单独看效果
  3. v1/v2 分别看效果
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
    ohlcv_cache[key] = df.set_index('dt').sort_index()
    print('done')
    return ohlcv_cache[key]

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
    fr_cache[symbol] = df.set_index('dt').sort_index()
    print(f'done ({len(df)} 条)')
    return fr_cache[symbol]

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
    if len(pnl_s) == 0 or pnl_s.isna().all():
        return dict(total=0, win_rate=0, rr=0, max_dd=0, sharpe=0, n=0)
    pnl_s = pnl_s.dropna()
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

def apply_short_filter(df, fr_col, threshold):
    """空头过滤：FR > threshold 才允许做空，多头不受影响"""
    short_mask = (df['direction'] == 'short') & (df[fr_col].notna()) & (df[fr_col] > threshold)
    long_mask  = df['direction'] == 'long'
    keep       = long_mask | short_mask
    return df[keep]['pnl_atr'], short_mask[df['direction'] == 'short'].mean()

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== 资金费率空头过滤 IS/OOS + 各标的拆分 ===\n")

    # 加载交易记录
    print("加载数据...")
    all_rows = []
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if df.empty: continue
            df['ver'] = ver; df['sym'] = sym; df['ccxt_sym'] = ccxt_sym
            all_rows.append(df)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    for sym in symbols:
        fetch_ohlcv(sym, '2h')
        fetch_funding_rate(sym)

    # 附上 ATR 定仓和资金费率
    enriched = []
    for df in all_rows:
        ccxt_sym = df['ccxt_sym'].iloc[0]
        ohlcv    = ohlcv_cache[(ccxt_sym, '2h')]
        fr_df    = fr_cache[ccxt_sym]
        df = df.copy()
        atr_a        = alloc_atr(ohlcv)
        df['alloc']  = merge_val(df, atr_a).values
        df['pnl_atr']= df['pnl'] * (df['alloc'] / BASE_CAPITAL)
        df['fr']     = merge_val(df, fr_df['fr']).values
        df['fr_ma3'] = merge_val(df, fr_df['fr'].rolling(3).mean()).values
        df['fr_ma7'] = merge_val(df, fr_df['fr'].rolling(7).mean()).values
        enriched.append(df)

    all_df = pd.concat(enriched, ignore_index=True)

    # IS/OOS 切割点（按时间排序后取前 80%）
    all_df_sorted = all_df.sort_values('entry_dt').reset_index(drop=True)
    cutoff_idx    = int(len(all_df_sorted) * 0.8)
    cutoff_dt     = all_df_sorted.loc[cutoff_idx, 'entry_dt']
    is_mask       = all_df_sorted['entry_dt'] < cutoff_dt
    oos_mask      = ~is_mask

    is_df  = all_df_sorted[is_mask].copy()
    oos_df = all_df_sorted[oos_mask].copy()
    is_years  = (cutoff_dt - all_df_sorted['entry_dt'].min()).days / 365
    oos_years = (all_df_sorted['entry_dt'].max() - cutoff_dt).days / 365

    print(f"IS 截止：{cutoff_dt.date()}（{is_years:.1f}年，{len(is_df)}笔）")
    print(f"OOS 起始：{cutoff_dt.date()}（{oos_years:.1f}年，{len(oos_df)}笔）\n")

    # 参数：最优 FR_MA7 > 0，也顺带看 FR > 0
    test_configs = [
        ('原始FR > 0',   'fr',     0.0),
        ('FR_MA3 > 0',  'fr_ma3', 0.0),
        ('FR_MA7 > 0',  'fr_ma7', 0.0),
        ('FR_MA7 > -0.010%', 'fr_ma7', -0.0001),
    ]

    # ── 1. IS/OOS 验证 ────────────────────────────────────────────────────────
    print("=" * 70)
    print("1. IS / OOS 验证（全标的合并）")
    print("=" * 70)

    s_base_is  = stats(is_df['pnl_atr'],  n_years=is_years,  n_strats=8)
    s_base_oos = stats(oos_df['pnl_atr'], n_years=oos_years, n_strats=8)
    print(f"\n基准（ATR定仓，无过滤）：")
    print(f"  IS ：Sharpe={s_base_is['sharpe']:.3f}  总PnL={s_base_is['total']:+.0f}  "
          f"最大回撤={s_base_is['max_dd']:.1f}%  ({len(is_df)}笔)")
    print(f"  OOS：Sharpe={s_base_oos['sharpe']:.3f}  总PnL={s_base_oos['total']:+.0f}  "
          f"最大回撤={s_base_oos['max_dd']:.1f}%  ({len(oos_df)}笔)")

    print(f"\n{'过滤条件':>20s}  {'IS Sharpe':>10s}  {'IS vs基准':>10s}  "
          f"{'OOS Sharpe':>11s}  {'OOS vs基准':>11s}  {'空头保留IS':>10s}  {'空头保留OOS':>11s}")
    print("-" * 88)
    for label, fr_col, thresh in test_configs:
        pnl_is,  kr_is  = apply_short_filter(is_df,  fr_col, thresh)
        pnl_oos, kr_oos = apply_short_filter(oos_df, fr_col, thresh)
        si  = stats(pnl_is,  n_years=is_years,  n_strats=8)
        so  = stats(pnl_oos, n_years=oos_years, n_strats=8)
        print(f"{label:>20s}  {si['sharpe']:>10.3f}  {si['sharpe']-s_base_is['sharpe']:>+10.3f}  "
              f"{so['sharpe']:>11.3f}  {so['sharpe']-s_base_oos['sharpe']:>+11.3f}  "
              f"{kr_is*100:>9.1f}%  {kr_oos*100:>10.1f}%")

    # ── 2. 各标的拆分（全量数据，FR_MA7 > 0）─────────────────────────────────
    print(f"\n{'='*70}")
    print("2. 各标的拆分（全量 2019-2026，FR_MA7 > 0）")
    print("=" * 70)

    # 按标的分组，每个标的用自己的 n_strats=2（v1+v2 各自独立）
    print(f"\n{'标的':>6s}  {'版本':>4s}  {'笔数':>5s}  "
          f"{'基准Sharpe':>10s}  {'过滤Sharpe':>10s}  {'差值':>7s}  "
          f"{'基准总PnL':>9s}  {'过滤总PnL':>9s}  {'空头保留':>8s}")
    print("-" * 85)

    sym_ver_results = []
    for ver in ['v1', 'v2']:
        for sym in ['BTC', 'ETH', 'SOL', 'DOGE']:
            sub = all_df[(all_df['ver'] == ver) & (all_df['sym'] == sym)].copy()
            if len(sub) == 0: continue
            n_yrs = (sub['entry_dt'].max() - sub['entry_dt'].min()).days / 365
            if n_yrs < 0.1: continue

            s_b = stats(sub['pnl_atr'], n_years=n_yrs, n_strats=1)

            # FR_MA7 > 0 过滤
            short_mask = (sub['direction'] == 'short') & sub['fr_ma7'].notna() & (sub['fr_ma7'] > 0)
            long_mask  = sub['direction'] == 'long'
            filtered_pnl = sub[long_mask | short_mask]['pnl_atr']
            keep_rate    = short_mask[sub['direction'] == 'short'].mean()
            s_f = stats(filtered_pnl, n_years=n_yrs, n_strats=1)

            diff = s_f['sharpe'] - s_b['sharpe']
            sym_ver_results.append(dict(ver=ver, sym=sym, n=len(sub),
                                        base_sharpe=s_b['sharpe'], filt_sharpe=s_f['sharpe'],
                                        diff=diff, base_total=s_b['total'], filt_total=s_f['total'],
                                        keep_rate=keep_rate))
            marker = ' ◀' if diff > 0 else ''
            print(f"{sym:>6s}  {ver:>4s}  {len(sub):>5d}  "
                  f"{s_b['sharpe']:>10.3f}  {s_f['sharpe']:>10.3f}  {diff:>+7.3f}  "
                  f"{s_b['total']:>+9.0f}  {s_f['total']:>+9.0f}  {keep_rate*100:>7.1f}%{marker}")

    # 按标的汇总（v1+v2 合并）
    print(f"\n{'标的汇总（v1+v2）':>6s}")
    print(f"{'标的':>6s}  {'笔数':>5s}  {'基准Sharpe':>10s}  {'过滤Sharpe':>10s}  "
          f"{'差值':>7s}  {'基准总PnL':>9s}  {'过滤总PnL':>9s}  {'空头保留':>8s}")
    print("-" * 75)
    for sym in ['BTC', 'ETH', 'SOL', 'DOGE']:
        sub = all_df[all_df['sym'] == sym].copy()
        if len(sub) == 0: continue
        n_yrs = (sub['entry_dt'].max() - sub['entry_dt'].min()).days / 365
        s_b = stats(sub['pnl_atr'], n_years=n_yrs, n_strats=2)
        short_mask = (sub['direction'] == 'short') & sub['fr_ma7'].notna() & (sub['fr_ma7'] > 0)
        long_mask  = sub['direction'] == 'long'
        filtered_pnl = sub[long_mask | short_mask]['pnl_atr']
        keep_rate    = short_mask[sub['direction'] == 'short'].mean()
        s_f = stats(filtered_pnl, n_years=n_yrs, n_strats=2)
        diff = s_f['sharpe'] - s_b['sharpe']
        marker = ' ◀' if diff > 0 else ''
        print(f"{sym:>6s}  {len(sub):>5d}  {s_b['sharpe']:>10.3f}  {s_f['sharpe']:>10.3f}  "
              f"{diff:>+7.3f}  {s_b['total']:>+9.0f}  {s_f['total']:>+9.0f}  "
              f"{keep_rate*100:>7.1f}%{marker}")

    # ── 3. IS/OOS × 各标的交叉（核心稳健性检验）────────────────────────────
    print(f"\n{'='*70}")
    print("3. IS/OOS × 各标的（FR_MA7 > 0，稳健性检验）")
    print("=" * 70)
    print(f"\n{'标的':>6s}  {'IS基准':>8s}  {'IS过滤':>8s}  {'IS差值':>8s}  "
          f"{'OOS基准':>9s}  {'OOS过滤':>9s}  {'OOS差值':>9s}")
    print("-" * 66)

    n_positive_oos = 0
    n_total = 0
    for sym in ['BTC', 'ETH', 'SOL', 'DOGE']:
        for subset_label, subset_df, n_yrs_s in [
            ('IS', is_df, is_years), ('OOS', oos_df, oos_years)
        ]:
            pass  # 下面统一处理

        sub_is  = is_df[is_df['sym'] == sym].copy()
        sub_oos = oos_df[oos_df['sym'] == sym].copy()
        if len(sub_is) < 5 or len(sub_oos) < 5: continue

        is_yrs_sym  = (sub_is['entry_dt'].max()  - sub_is['entry_dt'].min()).days  / 365
        oos_yrs_sym = (sub_oos['entry_dt'].max() - sub_oos['entry_dt'].min()).days / 365

        def sym_filter(sdf):
            sm = (sdf['direction'] == 'short') & sdf['fr_ma7'].notna() & (sdf['fr_ma7'] > 0)
            lm = sdf['direction'] == 'long'
            return sdf[lm | sm]['pnl_atr']

        s_b_is  = stats(sub_is['pnl_atr'],   n_years=max(is_yrs_sym, 0.1),  n_strats=2)
        s_f_is  = stats(sym_filter(sub_is),   n_years=max(is_yrs_sym, 0.1),  n_strats=2)
        s_b_oos = stats(sub_oos['pnl_atr'],  n_years=max(oos_yrs_sym, 0.1), n_strats=2)
        s_f_oos = stats(sym_filter(sub_oos), n_years=max(oos_yrs_sym, 0.1), n_strats=2)

        d_is  = s_f_is['sharpe']  - s_b_is['sharpe']
        d_oos = s_f_oos['sharpe'] - s_b_oos['sharpe']
        n_total += 1
        if d_oos > 0: n_positive_oos += 1

        print(f"{sym:>6s}  {s_b_is['sharpe']:>8.3f}  {s_f_is['sharpe']:>8.3f}  {d_is:>+8.3f}  "
              f"{s_b_oos['sharpe']:>9.3f}  {s_f_oos['sharpe']:>9.3f}  {d_oos:>+9.3f}")

    print(f"\nOOS 正向标的：{n_positive_oos}/{n_total}")

    # ── 4. 结论总结 ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("4. 结论总结")
    print("=" * 70)
    # 重算全量最优
    pnl_full, kr_full = apply_short_filter(all_df, 'fr_ma7', 0.0)
    s_full_base = stats(all_df['pnl_atr'], n_years=7, n_strats=8)
    s_full_filt = stats(pnl_full, n_years=7, n_strats=8)
    pnl_is_f,  _ = apply_short_filter(is_df,  'fr_ma7', 0.0)
    pnl_oos_f, _ = apply_short_filter(oos_df, 'fr_ma7', 0.0)
    si_f = stats(pnl_is_f,  n_years=is_years,  n_strats=8)
    so_f = stats(pnl_oos_f, n_years=oos_years, n_strats=8)

    print(f"\n  最优条件：FR_MA7 > 0（56H 资金费率均值为正才允许做空）")
    print(f"  空头保留率：{kr_full*100:.1f}%（排除 {(1-kr_full)*100:.1f}% 的空头交易）")
    print(f"\n  全量：{s_full_base['sharpe']:.3f} → {s_full_filt['sharpe']:.3f}  "
          f"({s_full_filt['sharpe']-s_full_base['sharpe']:+.3f})")
    print(f"  IS  ：{s_base_is['sharpe']:.3f} → {si_f['sharpe']:.3f}  "
          f"({si_f['sharpe']-s_base_is['sharpe']:+.3f})")
    print(f"  OOS ：{s_base_oos['sharpe']:.3f} → {so_f['sharpe']:.3f}  "
          f"({so_f['sharpe']-s_base_oos['sharpe']:+.3f})")
    print(f"  OOS/IS 改善比：{(so_f['sharpe']-s_base_oos['sharpe'])/(si_f['sharpe']-s_base_is['sharpe']):.2f}x")

if __name__ == '__main__':
    run()
