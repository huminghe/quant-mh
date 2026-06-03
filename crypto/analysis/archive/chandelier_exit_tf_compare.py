"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
Chandelier Exit 多时间框架对比验证（2026-06-03）

在 2H / 4H / 8H 三个 OHLCV 时间框架上分别计算 CE 止损线，
出场价格统一用 8H 收盘价（策略实际执行框架）。
对每个框架做 IS/OOS 验证，最终并排对比。

逻辑：
  - CE 止损线来自对应时间框架的 OHLCV
  - 触发判断：当该时间框架的 bar close 穿越止损线时触发
  - 出场价格：触发 bar 之后的第一根 8H bar 收盘价
    （更细粒度触发，8H 执行，模拟实盘可行性）

参数固定为 IS 最优附近的合理区间：
  ce_period=[14, 20], atr_period=[14, 20], k=[3.0, 3.5]
  共 8 组参数 × 3 时间框架 = 24 组
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

CE_PERIODS  = [14, 20]
ATR_PERIODS = [14, 20]
K_MULTS     = [3.0, 3.5]
TF_LIST     = ['2h', '4h', '8h']

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
def fetch_ohlcv(symbol, tf):
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
    return hh - k * atr, ll + k * atr   # long_stop, short_stop

# ─── 单笔模拟 ─────────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price):
    if entry_price <= 0 or exit_price <= 0: return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * BASE_CAPITAL
    return gross - BASE_CAPITAL * COMMISSION * 2

def simulate_trade_tf(trade, ce_ohlcv, exec_ohlcv_8h, long_stop_s, short_stop_s):
    """
    ce_ohlcv：用于判断 CE 止损触发的 OHLCV（2H/4H/8H）
    exec_ohlcv_8h：用于获取出场执行价的 8H OHLCV
    触发：ce_ohlcv 中 close 穿越止损线
    出场价：触发 bar 之后第一根 8H bar 的收盘价
    """
    entry_dt  = trade['entry_dt']
    exit_dt   = trade['exit_dt']
    direction = trade['direction']

    mask   = (ce_ohlcv.index > entry_dt) & (ce_ohlcv.index < exit_dt)
    window = ce_ohlcv[mask]
    if window.empty:
        return trade['pnl'], False

    stop_s = long_stop_s if direction == 1 else short_stop_s

    for dt, row in window.iterrows():
        stop_val = stop_s.loc[dt] if dt in stop_s.index else np.nan
        if np.isnan(stop_val): continue
        triggered = (direction == 1 and row['close'] < stop_val) or \
                    (direction == -1 and row['close'] > stop_val)
        if triggered:
            # 找触发后第一根 8H bar（含当根）
            exec_mask = exec_ohlcv_8h.index >= dt
            exec_bars = exec_ohlcv_8h[exec_mask]
            if exec_bars.empty:
                return trade['pnl'], False
            exec_price = exec_bars.iloc[0]['close']
            new_pnl = recalc_pnl(direction, trade['entry_price'], exec_price)
            return (trade['pnl'] if np.isnan(new_pnl) else new_pnl), True

    return trade['pnl'], False

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years, n_combos=8):
    if len(pnl_series) < 5:
        return dict(total=0, wr=0, rr=0, max_dd=0, sharpe=0, n=len(pnl_series))
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

def run_ce_subset(trades_dict, ce_ohlcv_map, exec_ohlcv_map, ce_cache, ce_p, atr_p, k, tf):
    all_pnls = []
    for (ver, sym), (trades, ccxt_sym) in trades_dict.items():
        if trades.empty: continue
        ce_ohlcv   = ce_ohlcv_map[(ccxt_sym, tf)]
        exec_ohlcv = exec_ohlcv_map[ccxt_sym]  # 8H
        ls, ss = ce_cache[(ccxt_sym, tf, ce_p, atr_p, k)]
        for _, row in trades.iterrows():
            p, _ = simulate_trade_tf(row, ce_ohlcv, exec_ohlcv, ls, ss)
            all_pnls.append(p)
    return pd.Series(all_pnls)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== Chandelier Exit 多时间框架对比（2H / 4H / 8H）===")
    print("CE 止损判断：对应 TF 的 close；出场执行价：8H 收盘价")
    print(f"参数：ce_period={CE_PERIODS}, atr_period={ATR_PERIODS}, k={K_MULTS}")
    n_params = len(CE_PERIODS) * len(ATR_PERIODS) * len(K_MULTS)
    print(f"每个 TF 参数组合数：{n_params}，共 {n_params * len(TF_LIST)} 组\n")

    # 加载交易记录
    print("加载交易记录...")
    all_trades_full = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades_full[(ver, sym)] = (trades, ccxt_sym)

    # IS/OOS 分割
    all_trades_is, all_trades_oos = {}, {}
    for key, (trades, ccxt_sym) in all_trades_full.items():
        n    = len(trades)
        n_is = int(n * 0.8)
        all_trades_is[key]  = (trades.iloc[:n_is].reset_index(drop=True), ccxt_sym)
        all_trades_oos[key] = (trades.iloc[n_is:].reset_index(drop=True), ccxt_sym)

    # 拉取 OHLCV（2H/4H/8H）
    symbols = list({ccxt_sym for _, (_, ccxt_sym) in all_trades_full.items()})
    print("\n拉取 OHLCV 数据...")
    ce_ohlcv_map   = {}  # (sym, tf) → ohlcv
    exec_ohlcv_map = {}  # sym → 8H ohlcv
    for sym in symbols:
        for tf in TF_LIST:
            ce_ohlcv_map[(sym, tf)] = fetch_ohlcv(sym, tf)
        exec_ohlcv_map[sym] = ce_ohlcv_map[(sym, '8h')]

    # 预计算 CE 指标
    param_combos = list(product(CE_PERIODS, ATR_PERIODS, K_MULTS))
    ce_cache = {}
    print(f"\n预计算 CE 指标...", end=' ', flush=True)
    for sym in symbols:
        for tf in TF_LIST:
            ohlcv = ce_ohlcv_map[(sym, tf)]
            for ce_p, atr_p, k in param_combos:
                ce_cache[(sym, tf, ce_p, atr_p, k)] = calc_chandelier(ohlcv, ce_p, atr_p, k)
    print("done")

    IS_YEARS, OOS_YEARS = 5.6, 1.4

    # 基准
    is_base_pnl  = pd.concat([t for t, _ in all_trades_is.values()])['pnl']
    oos_base_pnl = pd.concat([t for t, _ in all_trades_oos.values()])['pnl']
    is_base  = calc_stats(is_base_pnl,  IS_YEARS)
    oos_base = calc_stats(oos_base_pnl, OOS_YEARS)

    print(f"\n基准（原始出场）：IS Sharpe={is_base['sharpe']:.3f}  OOS Sharpe={oos_base['sharpe']:.3f}")
    print(f"IS N={is_base['n']}，OOS N={oos_base['n']}")

    # 各 TF 参数扫描
    tf_summary = {}  # tf → (best_params, is_sharpe, oos_sharpe, oos_diff, top5_oos_all_pos)

    for tf in TF_LIST:
        print(f"\n{'─'*72}")
        print(f"TF = {tf}")

        # IS 扫描
        is_results = []
        for ce_p, atr_p, k in param_combos:
            pnl_s = run_ce_subset(all_trades_is, ce_ohlcv_map, exec_ohlcv_map,
                                   ce_cache, ce_p, atr_p, k, tf)
            st = calc_stats(pnl_s, IS_YEARS)
            is_results.append({
                'ce_p': ce_p, 'atr_p': atr_p, 'k': k,
                'is_sharpe': st['sharpe'],
                'is_diff':   st['sharpe'] - is_base['sharpe'],
            })
        is_df = pd.DataFrame(is_results).sort_values('is_sharpe', ascending=False)
        best  = is_df.iloc[0]
        best_params = (int(best['ce_p']), int(best['atr_p']), best['k'])

        print(f"IS 最优：ce={best_params[0]}, atr={best_params[1]}, k={best_params[2]:.1f}  "
              f"IS Sharpe={best['is_sharpe']:.3f} ({'+' if best['is_diff']>=0 else ''}{best['is_diff']:.3f})")

        # IS Top 5 在 OOS 的稳健性
        oos_diffs = []
        print(f"  IS Top 5 OOS 稳健性：")
        for _, r in is_df.head(5).iterrows():
            oos_s = run_ce_subset(all_trades_oos, ce_ohlcv_map, exec_ohlcv_map,
                                   ce_cache, int(r['ce_p']), int(r['atr_p']), r['k'], tf)
            oos_st  = calc_stats(oos_s, OOS_YEARS)
            oos_d   = oos_st['sharpe'] - oos_base['sharpe']
            oos_diffs.append(oos_d)
            robust  = "YES" if oos_d > 0 else "NO"
            print(f"    ce={int(r['ce_p'])}, atr={int(r['atr_p'])}, k={r['k']:.1f}  "
                  f"IS {'+' if r['is_diff']>=0 else ''}{r['is_diff']:.3f}  "
                  f"OOS {'+' if oos_d>=0 else ''}{oos_d:.3f}  [{robust}]")

        # IS 最优参数的 OOS 结果
        best_oos_s  = run_ce_subset(all_trades_oos, ce_ohlcv_map, exec_ohlcv_map,
                                     ce_cache, *best_params, tf)
        best_oos_st = calc_stats(best_oos_s, OOS_YEARS)
        oos_diff    = best_oos_st['sharpe'] - oos_base['sharpe']
        top5_robust = sum(1 for d in oos_diffs if d > 0)

        print(f"IS 最优 → OOS Sharpe={best_oos_st['sharpe']:.3f} ({'+' if oos_diff>=0 else ''}{oos_diff:.3f})  "
              f"Top5 稳健 {top5_robust}/5")

        tf_summary[tf] = {
            'best_params':  best_params,
            'is_sharpe':    best['is_sharpe'],
            'is_diff':      best['is_diff'],
            'oos_sharpe':   best_oos_st['sharpe'],
            'oos_diff':     oos_diff,
            'top5_robust':  top5_robust,
        }

    # 汇总对比表
    print(f"\n{'═'*72}")
    print(f"汇总对比（基准：IS={is_base['sharpe']:.3f}，OOS={oos_base['sharpe']:.3f}）")
    print(f"{'═'*72}")
    print(f"{'TF':>4} {'最优参数':>22} {'IS Sharpe':>10} {'IS diff':>8} {'OOS Sharpe':>11} {'OOS diff':>9} {'Top5':>5}")
    print('─' * 72)
    for tf in TF_LIST:
        s = tf_summary[tf]
        ce_p, atr_p, k = s['best_params']
        is_sign  = '+' if s['is_diff']  >= 0 else ''
        oos_sign = '+' if s['oos_diff'] >= 0 else ''
        print(f"{tf:>4}  ce={ce_p:2d},atr={atr_p:2d},k={k:.1f}  "
              f"{s['is_sharpe']:>10.3f} {is_sign}{s['is_diff']:>7.3f}  "
              f"{s['oos_sharpe']:>10.3f} {oos_sign}{s['oos_diff']:>8.3f}  "
              f"{s['top5_robust']:>3}/5")

    print(f"\n判断依据：OOS diff 越大且 Top5 稳健数越多越好。")

if __name__ == '__main__':
    run()
