"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
Chandelier Exit 样本外验证（2026-06-03）

基于 chandelier_exit_validation.py 的结果，做 IS/OOS 分割验证。

流程：
  1. 按时间排序，前 80% 交易为样本内（IS），后 20% 为样本外（OOS）
  2. 在 IS 上扫描全部 60 组参数，选出最优
  3. 用 IS 最优参数在 OOS 上验证
  4. 报告 IS Sharpe、OOS Sharpe、IS/OOS 比值

IS/OOS 分割方式：
  - 按每个标的×版本组合各自的时间分割（不跨标的混排）
  - 分割点为各自交易序列的第 80% 条

时间框架：8H（Binance 永续，479m 策略）
标的：BTC/ETH/SOL/DOGE，v1 + v2
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path
from itertools import product

BASE_CAPITAL = 10_000
COMMISSION   = 0.0008
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

CE_PERIODS  = [10, 14, 20, 30]
ATR_PERIODS = [10, 14, 20]
K_MULTS     = [1.5, 2.0, 2.5, 3.0, 3.5]

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_trades(fname):
    path = downloads / fname
    if not path.exists():
        print(f'  ⚠ 找不到文件：{fname}')
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {name: i for i, name in enumerate(rows[0]) if name is not None}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx.get('交易 #', col_idx.get('Trade #', -1))]
            typ = str(row[col_idx.get('类型', col_idx.get('Type', -1))])
            dt  = row[col_idx.get('日期和时间', col_idx.get('Date/Time', -1))]
            pnl_key   = '净损益 USDT' if '净损益 USDT' in col_idx else 'Profit USDT'
            price_key = next((k for k in col_idx if '价格' in k or k == 'Price'), None)
            pnl   = row[col_idx.get(pnl_key, -1)]
            price = row[col_idx[price_key]] if price_key else None
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt']    = pd.Timestamp(dt)
                by_num[num]['entry_price'] = float(price) if price is not None else np.nan
                by_num[num]['direction']   = 1 if ('多' in typ or 'Long' in typ) else -1
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['exit_dt']    = pd.Timestamp(dt)
                by_num[num]['exit_price'] = float(price) if price is not None else np.nan
                by_num[num]['pnl']        = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values()
                if all(k in d for k in ('entry_dt','exit_dt','entry_price','exit_price','pnl','direction'))]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    df['exit_dt']  = (pd.to_datetime(df['exit_dt'])  - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol):
    if symbol in ohlcv_cache: return ohlcv_cache[symbol]
    print(f'  拉取 {symbol} 8H...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, '8h', since=since, limit=1000)
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
    ohlcv_cache[symbol] = df
    print('done')
    return df

# ─── CE 指标 ──────────────────────────────────────────────────────────────────

def calc_chandelier(ohlcv_df, ce_period, atr_period, k):
    high, low, close = ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr        = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    hh         = high.rolling(ce_period, min_periods=1).max()
    ll         = low.rolling(ce_period,  min_periods=1).min()
    long_stop  = hh - k * atr
    short_stop = ll + k * atr
    return long_stop, short_stop

# ─── 单笔模拟 ─────────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price):
    if entry_price <= 0 or exit_price <= 0: return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * BASE_CAPITAL
    cost  = BASE_CAPITAL * COMMISSION * 2
    return gross - cost

def simulate_trade(trade, ohlcv_df, long_stop_s, short_stop_s):
    entry_dt  = trade['entry_dt']
    exit_dt   = trade['exit_dt']
    direction = trade['direction']
    mask   = (ohlcv_df.index > entry_dt) & (ohlcv_df.index < exit_dt)
    window = ohlcv_df[mask]
    if window.empty:
        return trade['pnl'], False
    stop_s = long_stop_s if direction == 1 else short_stop_s
    for dt, row in window.iterrows():
        stop_val = stop_s.loc[dt] if dt in stop_s.index else np.nan
        if np.isnan(stop_val): continue
        triggered = (direction == 1 and row['close'] < stop_val) or \
                    (direction == -1 and row['close'] > stop_val)
        if triggered:
            new_pnl = recalc_pnl(direction, trade['entry_price'], row['close'])
            return (trade['pnl'] if np.isnan(new_pnl) else new_pnl), True
    return trade['pnl'], False

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years=None, n_combos=8):
    if len(pnl_series) < 5:
        return dict(total=0, wr=0, rr=0, max_dd=0, sharpe=0, n=len(pnl_series))
    if n_years is None:
        n_years = max(len(pnl_series) / (7867 / 7), 0.5)  # 按比例估算年数
    total  = pnl_series.sum()
    wr     = (pnl_series > 0).mean() * 100
    wins   = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    rr     = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 0
    cum    = pnl_series.cumsum()
    dd     = ((cum - cum.cummax()) / (BASE_CAPITAL * n_combos) * 100).min()
    sharpe = ((pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series) / n_years)
              if pnl_series.std() > 0 else 0)
    return dict(total=total, wr=wr, rr=rr, max_dd=dd, sharpe=sharpe, n=len(pnl_series))

# ─── 对单个数据集（IS 或 OOS）跑 CE 模拟 ─────────────────────────────────────

def run_ce_on_subset(trades_dict, all_ohlcv, ce_cache, ce_p, atr_p, k):
    """给定参数，对 trades_dict 中所有标的跑 CE，返回合并后的 pnl Series"""
    all_pnls = []
    for (ver, sym), (trades, ccxt_sym) in trades_dict.items():
        if trades.empty: continue
        ohlcv = all_ohlcv[ccxt_sym]
        long_stop, short_stop = ce_cache[(ccxt_sym, ce_p, atr_p, k)]
        for _, row in trades.iterrows():
            new_pnl, _ = simulate_trade(row, ohlcv, long_stop, short_stop)
            all_pnls.append(new_pnl)
    return pd.Series(all_pnls)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== Chandelier Exit 样本内/样本外验证 ===")
    print("IS/OOS 分割：前 80% 交易（按时间）为 IS，后 20% 为 OOS")
    print("流程：IS 选最优参数 → OOS 验证\n")

    # 加载
    print("加载交易记录...")
    all_trades_full = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades_full[(ver, sym)] = (trades, ccxt_sym)

    # IS/OOS 分割（每个标的×版本各自按时间分割）
    all_trades_is  = {}
    all_trades_oos = {}
    split_info = []
    for key, (trades, ccxt_sym) in all_trades_full.items():
        n     = len(trades)
        n_is  = int(n * 0.8)
        is_df  = trades.iloc[:n_is].reset_index(drop=True)
        oos_df = trades.iloc[n_is:].reset_index(drop=True)
        all_trades_is[key]  = (is_df,  ccxt_sym)
        all_trades_oos[key] = (oos_df, ccxt_sym)
        split_info.append({
            'key': f'{key[0]}_{key[1]}',
            'total': n, 'n_is': n_is, 'n_oos': n - n_is,
            'split_date': is_df['exit_dt'].iloc[-1].strftime('%Y-%m-%d'),
        })

    print("\nIS/OOS 分割点：")
    for s in split_info:
        print(f"  {s['key']:<12} 总 {s['total']:4d} 笔 | IS {s['n_is']:4d} (前80%) | "
              f"OOS {s['n_oos']:3d} (后20%) | 分割日期 {s['split_date']}")

    # OHLCV
    print("\n拉取 OHLCV 数据...")
    all_ohlcv = {}
    for (ver, sym), (trades, ccxt_sym) in all_trades_full.items():
        if ccxt_sym not in all_ohlcv:
            all_ohlcv[ccxt_sym] = fetch_ohlcv(ccxt_sym)

    # 预计算 CE 指标
    param_combos = list(product(CE_PERIODS, ATR_PERIODS, K_MULTS))
    ce_cache = {}
    print(f"\n预计算 CE 指标...", end=' ', flush=True)
    for ccxt_sym, ohlcv in all_ohlcv.items():
        for ce_p, atr_p, k in param_combos:
            ce_cache[(ccxt_sym, ce_p, atr_p, k)] = calc_chandelier(ohlcv, ce_p, atr_p, k)
    print("done")

    # 基准统计
    is_baseline_pnl  = pd.concat([t for t, _ in all_trades_is.values()])['pnl']
    oos_baseline_pnl = pd.concat([t for t, _ in all_trades_oos.values()])['pnl']

    n_is_years  = (is_baseline_pnl.index[-1] - is_baseline_pnl.index[0] + 1) / len(is_baseline_pnl)
    is_years    = 7 * 0.8  # ≈ 5.6 年
    oos_years   = 7 * 0.2  # ≈ 1.4 年

    is_base  = calc_stats(is_baseline_pnl,  n_years=is_years)
    oos_base = calc_stats(oos_baseline_pnl, n_years=oos_years)

    print(f"\n{'─'*72}")
    print(f"基准（原始出场）")
    print(f"  IS  Sharpe={is_base['sharpe']:.3f}  Total={is_base['total']:+,.0f}  N={is_base['n']}")
    print(f"  OOS Sharpe={oos_base['sharpe']:.3f}  Total={oos_base['total']:+,.0f}  N={oos_base['n']}")
    print(f"{'─'*72}")

    # IS 参数扫描
    print(f"\n扫描 IS 上 {len(param_combos)} 组参数...")
    is_results = []
    for ce_p, atr_p, k in param_combos:
        pnl_s = run_ce_on_subset(all_trades_is, all_ohlcv, ce_cache, ce_p, atr_p, k)
        st    = calc_stats(pnl_s, n_years=is_years)
        is_results.append({
            'ce_p': ce_p, 'atr_p': atr_p, 'k': k,
            'is_sharpe': round(st['sharpe'], 3),
            'is_diff':   round(st['sharpe'] - is_base['sharpe'], 3),
        })

    is_df_res = pd.DataFrame(is_results).sort_values('is_sharpe', ascending=False)

    print(f"\nIS Top 10（按 Sharpe）：")
    print(f"{'ce_p':>5} {'atr_p':>5} {'k':>4} {'IS Sharpe':>10} {'IS diff':>8}")
    print('─' * 40)
    for _, r in is_df_res.head(10).iterrows():
        sign = '+' if r['is_diff'] >= 0 else ''
        print(f"{int(r['ce_p']):>5} {int(r['atr_p']):>5} {r['k']:>4.1f}  "
              f"{r['is_sharpe']:>9.3f}  {sign}{r['is_diff']:>7.3f}")

    # OOS 验证（用 IS 最优参数）
    best_is = is_df_res.iloc[0]
    best_ce_p, best_atr_p, best_k = int(best_is['ce_p']), int(best_is['atr_p']), best_is['k']

    print(f"\n{'─'*72}")
    print(f"IS 最优参数：ce={best_ce_p}, atr={best_atr_p}, k={best_k:.1f}")
    print(f"{'─'*72}")

    oos_pnl = run_ce_on_subset(all_trades_oos, all_ohlcv, ce_cache, best_ce_p, best_atr_p, best_k)
    oos_ce  = calc_stats(oos_pnl, n_years=oos_years)

    is_ce_pnl = run_ce_on_subset(all_trades_is, all_ohlcv, ce_cache, best_ce_p, best_atr_p, best_k)
    is_ce     = calc_stats(is_ce_pnl, n_years=is_years)

    print(f"\n结果对比（IS 最优参数）：")
    print(f"{'':12} {'基准 Sharpe':>12} {'CE Sharpe':>10} {'差值':>8}")
    print('─' * 50)
    print(f"{'IS':12} {is_base['sharpe']:>12.3f} {is_ce['sharpe']:>10.3f} "
          f"  {'+' if is_ce['sharpe'] >= is_base['sharpe'] else ''}{is_ce['sharpe'] - is_base['sharpe']:>7.3f}")
    print(f"{'OOS':12} {oos_base['sharpe']:>12.3f} {oos_ce['sharpe']:>10.3f} "
          f"  {'+' if oos_ce['sharpe'] >= oos_base['sharpe'] else ''}{oos_ce['sharpe'] - oos_base['sharpe']:>7.3f}")

    is_ratio = is_ce['sharpe'] / is_base['sharpe'] if is_base['sharpe'] != 0 else 0
    oos_ratio = oos_ce['sharpe'] / oos_base['sharpe'] if oos_base['sharpe'] != 0 else 0
    print(f"\nIS  CE/基准 比值：{is_ratio:.3f}")
    print(f"OOS CE/基准 比值：{oos_ratio:.3f}")
    if oos_ratio < 0.5:
        print(f"⚠ OOS 比值 < 0.5，可能存在过拟合（IS 改善未能迁移到 OOS）")
    elif oos_ratio >= 1.0:
        print(f"OOS 改善稳健（CE 在样本外持续有效）")
    else:
        print(f"OOS 有部分改善但低于 IS，属正常衰减")

    # 额外：测试 IS Top 5 参数在 OOS 的表现（检验稳健性）
    print(f"\n{'─'*72}")
    print(f"IS Top 5 参数在 OOS 的稳健性验证：")
    print(f"{'ce_p':>5} {'atr_p':>5} {'k':>4} {'IS diff':>8} {'OOS diff':>9} {'OOS 稳健?':>10}")
    print('─' * 55)
    for _, r in is_df_res.head(5).iterrows():
        oos_pnl_i = run_ce_on_subset(all_trades_oos, all_ohlcv, ce_cache,
                                      int(r['ce_p']), int(r['atr_p']), r['k'])
        oos_st_i  = calc_stats(oos_pnl_i, n_years=oos_years)
        oos_diff  = oos_st_i['sharpe'] - oos_base['sharpe']
        robust    = "YES" if oos_diff > 0 else "NO"
        is_sign   = '+' if r['is_diff'] >= 0 else ''
        oos_sign  = '+' if oos_diff >= 0 else ''
        print(f"{int(r['ce_p']):>5} {int(r['atr_p']):>5} {r['k']:>4.1f}  "
              f"{is_sign}{r['is_diff']:>7.3f}  {oos_sign}{oos_diff:>8.3f}  {robust:>10}")

    # 按标的拆分（IS 最优参数）
    print(f"\n{'─'*72}")
    print(f"按标的拆分（IS 最优参数 ce={best_ce_p}, atr={best_atr_p}, k={best_k:.1f}）：")
    print(f"{'':12} {'IS基准':>8} {'IS_CE':>7} {'IS差':>6} | {'OOS基准':>8} {'OOS_CE':>7} {'OOS差':>6}")
    print('─' * 72)
    for key in sorted(all_trades_is.keys()):
        label = f"{key[0]}_{key[1]}"
        is_trades,  ccxt_sym = all_trades_is[key]
        oos_trades, _        = all_trades_oos[key]
        ohlcv      = all_ohlcv[ccxt_sym]
        ls, ss     = ce_cache[(ccxt_sym, best_ce_p, best_atr_p, best_k)]

        is_pnls, oos_pnls = [], []
        for _, row in is_trades.iterrows():
            p, _ = simulate_trade(row, ohlcv, ls, ss)
            is_pnls.append(p)
        for _, row in oos_trades.iterrows():
            p, _ = simulate_trade(row, ohlcv, ls, ss)
            oos_pnls.append(p)

        is_b  = calc_stats(is_trades['pnl'],    n_years=is_years,  n_combos=1)
        is_c  = calc_stats(pd.Series(is_pnls),  n_years=is_years,  n_combos=1)
        oos_b = calc_stats(oos_trades['pnl'],   n_years=oos_years, n_combos=1)
        oos_c = calc_stats(pd.Series(oos_pnls), n_years=oos_years, n_combos=1)

        is_d  = is_c['sharpe']  - is_b['sharpe']
        oos_d = oos_c['sharpe'] - oos_b['sharpe']
        print(f"  {label:<10} {is_b['sharpe']:>8.3f} {is_c['sharpe']:>7.3f} "
              f"{'+' if is_d >= 0 else ''}{is_d:>5.3f} | "
              f"{oos_b['sharpe']:>8.3f} {oos_c['sharpe']:>7.3f} "
              f"{'+' if oos_d >= 0 else ''}{oos_d:>5.3f}")

    print(f"\n总交易笔数：IS={is_base['n']}，OOS={oos_base['n']}")
    print(f"\n结论判断标准：")
    print(f"  OOS diff > 0 且 Top5 稳健 → CE 有效，可考虑实施")
    print(f"  OOS diff > 0 但 Top5 不稳健 → IS 过拟合，不应实施")
    print(f"  OOS diff < 0 → CE 无效")

if __name__ == '__main__':
    run()
