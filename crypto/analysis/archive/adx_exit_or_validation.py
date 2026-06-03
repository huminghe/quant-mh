"""
ADX 出场优化验证 v2 — 静态条件 vs 动态条件 单独验证（2026-06-04）

背景：
  v1（AND 条件）验证结果：54 组参数全部触发率 0%，AND 条件过严。
  本脚本分别验证两个条件单独使用：
    模式 S（静态）：ADX < threshold
    模式 D（动态）：ADX 从近 N 根峰值下降 > drop_pct%

每种模式同样测 选项A（等 EMA 重新入场）和 选项B（ADX 恢复立即重新入场）。

时间框架：8H（Binance 永续）
标的：BTC/ETH/SOL/DOGE，v1 + v2，共 7,867 笔交易

用法：
  python adx_exit_or_validation.py
  python adx_exit_or_validation.py --detail
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--detail', action='store_true')
args = parser.parse_args()

BASE_CAPITAL = 10_000
COMMISSION   = 0.0008

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

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

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
            # 列名可能是 '价格 USDT' 或 '价格' 或 'Price'
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
            except:
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

def calc_adx(ohlcv_df, period=14):
    high, low, close = ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr  = tr.ewm(alpha=1/period, adjust=False).mean()
    up   = high.diff(); down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    di_p = 100 * dm_p.ewm(alpha=1/period, adjust=False).mean() / atr
    di_m = 100 * dm_m.ewm(alpha=1/period, adjust=False).mean() / atr
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ─── 出场触发逻辑（三种模式）─────────────────────────────────────────────────

def find_early_exit(adx_s, entry_dt, exit_dt, mode, params):
    """
    mode='S'：ADX < static_thresh
    mode='D'：ADX < rolling_peak * (1 - drop_pct)，回望 lookback_n 根
    """
    mask   = (adx_s.index >= entry_dt) & (adx_s.index < exit_dt)
    window = adx_s[mask]
    if window.empty: return None

    if mode == 'S':
        thresh = params['static_thresh']
        hit = window[window < thresh]
        return hit.index[0] if not hit.empty else None

    else:  # mode == 'D'
        drop_pct   = params['drop_pct']
        lookback_n = params['lookback_n']
        rolling_peak = adx_s.rolling(lookback_n, min_periods=1).max()
        for dt, val in window.items():
            if val < rolling_peak.loc[dt] * (1 - drop_pct):
                return dt
        return None

def find_adx_recovery(adx_s, from_dt, recovery_thresh, exit_dt):
    mask = (adx_s.index > from_dt) & (adx_s.index <= exit_dt)
    window = adx_s[mask]
    recovered = window[window >= recovery_thresh]
    return recovered.index[0] if not recovered.empty else None

# ─── PnL 重算 ─────────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price, capital=BASE_CAPITAL):
    if entry_price <= 0 or exit_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
        return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * capital
    return gross - capital * COMMISSION * 2

# ─── 单笔交易模拟 ─────────────────────────────────────────────────────────────

def simulate_option_a(trade, ohlcv_df, adx_s, mode, params):
    early_dt = find_early_exit(adx_s, trade['entry_dt'], trade['exit_dt'], mode, params)
    if early_dt is None: return trade['pnl'], False
    try:
        early_close = ohlcv_df.loc[early_dt, 'close']
    except KeyError:
        return trade['pnl'], False
    new_pnl = recalc_pnl(trade['direction'], trade['entry_price'], early_close)
    return (new_pnl if not np.isnan(new_pnl) else trade['pnl']), (not np.isnan(new_pnl))

def simulate_option_b(trade, ohlcv_df, adx_s, mode, params):
    # 恢复阈值：静态模式用 static_thresh，动态模式用 ADX 25（合理默认）
    recovery_thresh = params.get('static_thresh', 25)

    early_dt = find_early_exit(adx_s, trade['entry_dt'], trade['exit_dt'], mode, params)
    if early_dt is None: return trade['pnl'], False
    try:
        early_close = ohlcv_df.loc[early_dt, 'close']
    except KeyError:
        return trade['pnl'], False
    pnl_1 = recalc_pnl(trade['direction'], trade['entry_price'], early_close)
    if np.isnan(pnl_1): return trade['pnl'], False

    re_entry_dt = find_adx_recovery(adx_s, early_dt, recovery_thresh, trade['exit_dt'])
    if re_entry_dt is None: return pnl_1, True
    try:
        re_close = ohlcv_df.loc[re_entry_dt, 'close']
    except KeyError:
        return pnl_1, True
    pnl_2 = recalc_pnl(trade['direction'], re_close, trade['exit_price'])
    return (pnl_1 + (pnl_2 if not np.isnan(pnl_2) else 0)), True

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years=7, n_combos=8):
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

# ─── 主流程 ───────────────────────────────────────────────────────────────────

# 模式S参数：仅静态阈值
STATIC_THRESHOLDS = [10, 15, 20, 25, 30]

# 模式D参数：仅动态下降
DROP_PCTS   = [0.10, 0.20, 0.30]
LOOKBACK_NS = [3, 5, 10]

def run():
    print("=== ADX 出场优化验证 v2 — 单条件版本 ===")
    print("模式S：ADX < 静态阈值（单独）")
    print("模式D：ADX 从 N 根峰值下降 > X%（单独）")
    print(f"S参数：阈值={STATIC_THRESHOLDS}（{len(STATIC_THRESHOLDS)}组）")
    print(f"D参数：下降%={DROP_PCTS} × 回望={LOOKBACK_NS}（{len(DROP_PCTS)*len(LOOKBACK_NS)}组）")
    total_combos = (len(STATIC_THRESHOLDS) + len(DROP_PCTS)*len(LOOKBACK_NS)) * 2
    print(f"总组合：{total_combos}（× 2 选项）\n")

    # 加载数据
    print("加载交易记录...")
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades[(ver, sym)] = (trades, ccxt_sym)

    print("\n拉取 OHLCV 数据...")
    all_ohlcv, all_adx = {}, {}
    for (ver, sym), (trades, ccxt_sym) in all_trades.items():
        if ccxt_sym not in all_ohlcv:
            ohlcv = fetch_ohlcv(ccxt_sym)
            all_ohlcv[ccxt_sym] = ohlcv
            all_adx[ccxt_sym]   = calc_adx(ohlcv)

    # 基准
    baseline_pnl = pd.concat([t for t, _ in all_trades.values()])['pnl']
    baseline = calc_stats(baseline_pnl)
    print(f"\n{'─'*72}")
    print(f"基准（无 ADX 出场）：Sharpe={baseline['sharpe']:.3f}  "
          f"Total={baseline['total']:+,.0f}  MaxDD={baseline['max_dd']:.1f}%  N={baseline['n']}")
    print(f"{'─'*72}")

    results = []

    def sweep(mode, param_list):
        for params in param_list:
            for opt_name, sim_fn in [('A', simulate_option_a), ('B', simulate_option_b)]:
                new_pnls, n_trig = [], 0
                for (ver, sym), (trades, ccxt_sym) in all_trades.items():
                    ohlcv = all_ohlcv[ccxt_sym]
                    adx_s = all_adx[ccxt_sym]
                    for _, row in trades.iterrows():
                        new_pnl, triggered = sim_fn(row, ohlcv, adx_s, mode, params)
                        new_pnls.append(new_pnl)
                        if triggered: n_trig += 1
                pnl_s = pd.Series(new_pnls)
                st    = calc_stats(pnl_s)
                trig_rate = n_trig / len(new_pnls) * 100
                row = {'mode': mode, 'option': opt_name, **params,
                       'sharpe': round(st['sharpe'], 3),
                       'total':  round(st['total'], 0),
                       'max_dd': round(st['max_dd'], 1),
                       'trig%':  round(trig_rate, 1)}
                results.append(row)

    # 模式S
    s_params = [{'static_thresh': t} for t in STATIC_THRESHOLDS]
    sweep('S', s_params)

    # 模式D
    d_params = [{'drop_pct': d, 'lookback_n': n, 'static_thresh': 25}
                for d, n in product(DROP_PCTS, LOOKBACK_NS)]
    sweep('D', d_params)

    results_df = pd.DataFrame(results)

    # ── 输出 ──────────────────────────────────────────────────────────────────
    for mode, mode_name in [('S', '静态：ADX < 阈值'), ('D', '动态：ADX 从峰值下降 > X%')]:
        print(f"\n{'━'*72}")
        print(f"模式 {mode}（{mode_name}）")
        for opt in ['A', 'B']:
            sub = results_df[(results_df['mode'] == mode) & (results_df['option'] == opt)]
            sub = sub.sort_values('sharpe', ascending=False)
            best  = sub.iloc[0]
            worst = sub.iloc[-1]
            opt_desc = '等 EMA 信号' if opt == 'A' else 'ADX 恢复重入'
            print(f"\n  选项{opt}（{opt_desc}）  基准={baseline['sharpe']:.3f}")
            print(f"  最好 Sharpe={best['sharpe']:.3f}  最差={worst['sharpe']:.3f}")

            if mode == 'S':
                hdr = f"  {'thresh':>7} {'sharpe':>7} {'total':>10} {'maxDD%':>7} {'trig%':>6}"
            else:
                hdr = f"  {'drop%':>6} {'lookback':>8} {'sharpe':>7} {'total':>10} {'maxDD%':>7} {'trig%':>6}"
            print(hdr)
            print(f"  {'─'*55}")
            for _, r in sub.iterrows():
                if mode == 'S':
                    print(f"  {r['static_thresh']:>7} {r['sharpe']:>7.3f} {r['total']:>+10,.0f} "
                          f"{r['max_dd']:>7.1f} {r['trig%']:>6.1f}")
                else:
                    dp = int(r['drop_pct'] * 100) if 'drop_pct' in r else '—'
                    lb = int(r['lookback_n']) if 'lookback_n' in r else '—'
                    print(f"  {dp:>6} {lb:>8} {r['sharpe']:>7.3f} {r['total']:>+10,.0f} "
                          f"{r['max_dd']:>7.1f} {r['trig%']:>6.1f}")

    # ── 详细标的视图 ──────────────────────────────────────────────────────────
    if args.detail:
        for mode in ['S', 'D']:
            for opt in ['A', 'B']:
                sub = results_df[(results_df['mode'] == mode) & (results_df['option'] == opt)]
                best_row = sub.sort_values('sharpe', ascending=False).iloc[0]
                if mode == 'S':
                    best_params = {'static_thresh': int(best_row['static_thresh'])}
                    desc = f"static={best_params['static_thresh']}"
                else:
                    best_params = {'drop_pct': best_row['drop_pct'],
                                   'lookback_n': int(best_row['lookback_n']),
                                   'static_thresh': 25}
                    desc = f"drop={int(best_row['drop_pct']*100)}%, lookback={int(best_row['lookback_n'])}"
                sim_fn = simulate_option_a if opt == 'A' else simulate_option_b

                print(f"\n{'─'*72}")
                print(f"模式{mode} 选项{opt} 最优（{desc}）")
                print(f"{'─'*72}")
                for (ver, sym), (trades, ccxt_sym) in sorted(all_trades.items()):
                    ohlcv = all_ohlcv[ccxt_sym]
                    adx_s = all_adx[ccxt_sym]
                    new_pnls, n_trig = [], 0
                    for _, row in trades.iterrows():
                        new_pnl, triggered = sim_fn(row, ohlcv, adx_s, mode, best_params)
                        new_pnls.append(new_pnl)
                        if triggered: n_trig += 1
                    orig_s = calc_stats(trades['pnl'], n_combos=1)
                    new_s  = calc_stats(pd.Series(new_pnls), n_combos=1)
                    diff   = new_s['sharpe'] - orig_s['sharpe']
                    sign   = '+' if diff >= 0 else ''
                    print(f"  {ver}_{sym:<5} Sharpe {orig_s['sharpe']:5.3f}→{new_s['sharpe']:5.3f} "
                          f"({sign}{diff:.3f})  Total {orig_s['total']:+8,.0f}→{new_s['total']:+8,.0f}  "
                          f"触发率 {n_trig/len(trades)*100:.1f}%")

    print(f"\n总交易笔数：{baseline['n']}  参数总组合：{len(results_df)}")

if __name__ == '__main__':
    run()
