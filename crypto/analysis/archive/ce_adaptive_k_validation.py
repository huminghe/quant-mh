"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
Chandelier Exit ATR 乘数自适应验证（2026-06-03）

验证：把 CE 的固定 k=3.5 换成根据入场时 ATR 百分位动态变化的 k，
是否能改善趋势跟踪策略表现。

核心逻辑：
  - 入场信号：沿用原始 TV EMA 策略信号（从 xlsx 读取）
  - CE 出场：same as chandelier_exit_validation.py
  - 动态 k：在每笔交易入场时，查当前 ATR 百分位，按分位选 k：
      ATR_pct < 25th → k = K_LOW   (低波动，止损更紧)
      25th ≤ ATR_pct < 75th → k = K_MID  (中性，与基准一致)
      ATR_pct ≥ 75th → k = K_HIGH  (高波动，止损更宽)
  - 百分位窗口：过去 N 根 8H bar（默认 252，约 3 个月）

基准：固定 k=3.5（与 CE 验证脚本结论一致）

参数网格：
  - percentile_window：126, 252, 504（约 6周/3月/6月）
  - K_LOW / K_MID / K_HIGH：多组组合
  - ce_period / atr_period：固定为最优值（ce=20, atr=20）

数据：TV 导出 xlsx（2026-05-22）+ Binance 8H OHLCV（2019-2026）
标的：BTC/ETH/SOL/DOGE，v1 + v2

用法：
  python ce_adaptive_k_validation.py
  python ce_adaptive_k_validation.py --detail
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--detail', action='store_true', help='按标的展示最优参数详细结果')
args = parser.parse_args()

BASE_CAPITAL = 10_000
COMMISSION   = 0.0008   # 万分之八（taker 含滑点），双边

# CE 固定参数（沿用验证最优值）
CE_PERIOD  = 20
ATR_PERIOD = 20

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

# ─── 参数网格 ──────────────────────────────────────────────────────────────────
# K_LOW / K_MID / K_HIGH 组合，MID 固定 3.5（与基准一致）
K_CONFIGS = [
    (2.5, 3.5, 4.5),
    (2.8, 3.5, 4.2),
    (3.0, 3.5, 4.0),
    (2.5, 3.5, 4.0),
    (2.8, 3.5, 4.5),
    (3.0, 3.5, 4.5),
]
PERCENTILE_WINDOWS = [126, 252, 504]   # 约 6 周 / 3 月 / 6 月

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载 TV 交易记录，UTC+8 → UTC"""
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
    """拉取 8H OHLCV（Binance 永续）"""
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

# ─── 指标预计算 ────────────────────────────────────────────────────────────────

def calc_atr(ohlcv_df, atr_period):
    """计算 ATR（Wilder 平滑）"""
    high  = ohlcv_df['high']
    low   = ohlcv_df['low']
    close = ohlcv_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/atr_period, adjust=False).mean()

def calc_atr_percentile(atr_series, window):
    """
    计算每个时间点的 ATR 百分位（基于过去 window 根 bar）。
    返回 0-100 的 Series。
    """
    def pct_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1] * 100
    return atr_series.rolling(window, min_periods=window//2).apply(pct_rank, raw=True)

def calc_chandelier_stops(ohlcv_df, ce_period, atr_period):
    """预计算 CE 止损线（不含 k，后续乘以动态 k）"""
    high  = ohlcv_df['high']
    low   = ohlcv_df['low']
    close = ohlcv_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    hh = high.rolling(ce_period, min_periods=1).max()
    ll = low.rolling(ce_period,  min_periods=1).min()
    # 返回 atr、hh、ll，让外部按 k 动态计算止损线
    return atr, hh, ll

# ─── 模拟单笔交易 ──────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price):
    if entry_price <= 0 or exit_price <= 0: return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * BASE_CAPITAL
    cost  = BASE_CAPITAL * COMMISSION * 2
    return gross - cost

def simulate_trade_adaptive(trade, ohlcv_df, atr_s, hh_s, ll_s, atr_pct_s,
                             k_low, k_mid, k_high):
    """
    在入场时查 ATR 百分位，选择动态 k，然后扫描持仓窗口内是否触发 CE 止损。
    """
    entry_dt  = trade['entry_dt']
    exit_dt   = trade['exit_dt']
    direction = trade['direction']

    # 入场时的 ATR 百分位 → 选 k
    # 取最近 bar（entry_dt 时或之前最近的 bar）
    valid_pct = atr_pct_s[atr_pct_s.index <= entry_dt]
    if valid_pct.empty or np.isnan(valid_pct.iloc[-1]):
        # 百分位未就绪（数据不足），使用中性 k
        k = k_mid
    else:
        pct = valid_pct.iloc[-1]
        if pct < 25:
            k = k_low
        elif pct >= 75:
            k = k_high
        else:
            k = k_mid

    # 按选定的 k 实时计算止损线
    long_stop_s  = hh_s - k * atr_s
    short_stop_s = ll_s + k * atr_s

    # 持仓期间扫描
    mask = (ohlcv_df.index > entry_dt) & (ohlcv_df.index < exit_dt)
    window = ohlcv_df[mask]
    if window.empty:
        return trade['pnl'], False, None, k

    stop_s = long_stop_s if direction == 1 else short_stop_s
    for dt, row in window.iterrows():
        stop_val = stop_s.loc[dt] if dt in stop_s.index else np.nan
        if np.isnan(stop_val): continue
        triggered = (direction == 1 and row['close'] < stop_val) or \
                    (direction == -1 and row['close'] > stop_val)
        if triggered:
            new_pnl = recalc_pnl(direction, trade['entry_price'], row['close'])
            if np.isnan(new_pnl):
                return trade['pnl'], False, None, k
            return new_pnl, True, dt, k

    return trade['pnl'], False, None, k

# ─── 汇总统计 ──────────────────────────────────────────────────────────────────

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

def run():
    print("=== Chandelier Exit ATR 乘数自适应验证 ===")
    print(f"CE 固定参数：ce_period={CE_PERIOD}, atr_period={ATR_PERIOD}")
    print(f"基准：固定 k=3.5")
    print(f"动态 k 分位：< 25th → k_low，25-75th → k_mid=3.5，≥ 75th → k_high")
    print(f"K 配置组合：{len(K_CONFIGS)} 组 × 百分位窗口 {PERCENTILE_WINDOWS}")
    print(f"总测试组合：{len(K_CONFIGS) * len(PERCENTILE_WINDOWS)}\n")

    # ── 加载交易记录 ──────────────────────────────────────────────────────────
    print("加载交易记录...")
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades[(ver, sym)] = (trades, ccxt_sym)

    if not all_trades:
        print("错误：没有找到任何交易记录文件，请检查 Downloads 目录")
        return

    loaded = [f'{v}_{s}' for v, s in all_trades]
    print(f"已加载：{loaded}")

    print("\n拉取 OHLCV 数据...")
    all_ohlcv = {}
    for (ver, sym), (trades, ccxt_sym) in all_trades.items():
        if ccxt_sym not in all_ohlcv:
            all_ohlcv[ccxt_sym] = fetch_ohlcv(ccxt_sym)

    # ── 基准统计（固定 k=3.5）────────────────────────────────────────────────
    # 复用 chandelier_exit_validation 的固定 k=3.5 基准（ce=20, atr=20, k=3.5）
    print("\n计算基准（固定 k=3.5, ce=20, atr=20）...")
    baseline_pnls = []
    base_cache = {}   # ccxt_sym → (atr_s, hh_s, ll_s)
    for ccxt_sym, ohlcv in all_ohlcv.items():
        atr_s, hh_s, ll_s = calc_chandelier_stops(ohlcv, CE_PERIOD, ATR_PERIOD)
        base_cache[ccxt_sym] = (atr_s, hh_s, ll_s)
        long_stop  = hh_s - 3.5 * atr_s
        short_stop = ll_s + 3.5 * atr_s
        for (ver, sym), (trades, sym2) in all_trades.items():
            if sym2 != ccxt_sym: continue
            for _, row in trades.iterrows():
                mask = (ohlcv.index > row['entry_dt']) & (ohlcv.index < row['exit_dt'])
                window = ohlcv[mask]
                pnl = row['pnl']
                if not window.empty:
                    stop_s = long_stop if row['direction'] == 1 else short_stop
                    for dt2, r2 in window.iterrows():
                        sv = stop_s.loc[dt2] if dt2 in stop_s.index else np.nan
                        if np.isnan(sv): continue
                        trig = (row['direction'] == 1 and r2['close'] < sv) or \
                               (row['direction'] == -1 and r2['close'] > sv)
                        if trig:
                            pnl = recalc_pnl(row['direction'], row['entry_price'], r2['close'])
                            if np.isnan(pnl): pnl = row['pnl']
                            break
                baseline_pnls.append(pnl)

    baseline = calc_stats(pd.Series(baseline_pnls))
    print(f"\n{'─'*72}")
    print(f"基准（固定 k=3.5）：Sharpe={baseline['sharpe']:.3f}  "
          f"Total={baseline['total']:+,.0f}  MaxDD={baseline['max_dd']:.1f}%  "
          f"WR={baseline['wr']:.1f}%  RR={baseline['rr']:.2f}  N={baseline['n']}")
    print(f"{'─'*72}")

    # ── 预计算各窗口下的 ATR 百分位 ──────────────────────────────────────────
    print("\n预计算 ATR 百分位序列...")
    atr_pct_cache = {}   # (ccxt_sym, window) → percentile Series
    for ccxt_sym, ohlcv in all_ohlcv.items():
        atr_s, _, _ = base_cache[ccxt_sym]
        for window in PERCENTILE_WINDOWS:
            key = (ccxt_sym, window)
            atr_pct_cache[key] = calc_atr_percentile(atr_s, window)
    print("done")

    # ── 参数扫描 ──────────────────────────────────────────────────────────────
    results = []
    param_combos = list(product(K_CONFIGS, PERCENTILE_WINDOWS))
    print(f"\n扫描 {len(param_combos)} 组参数...")

    for (k_low, k_mid, k_high), pct_win in param_combos:
        new_pnls = []
        n_triggered = 0
        k_low_used = k_mid_used = k_high_used = 0

        for (ver, sym), (trades, ccxt_sym) in all_trades.items():
            ohlcv = all_ohlcv[ccxt_sym]
            atr_s, hh_s, ll_s = base_cache[ccxt_sym]
            atr_pct_s = atr_pct_cache[(ccxt_sym, pct_win)]

            for _, row in trades.iterrows():
                new_pnl, triggered, _, k_used = simulate_trade_adaptive(
                    row, ohlcv, atr_s, hh_s, ll_s, atr_pct_s,
                    k_low, k_mid, k_high
                )
                new_pnls.append(new_pnl)
                if triggered: n_triggered += 1
                if k_used == k_low:  k_low_used  += 1
                elif k_used == k_high: k_high_used += 1
                else: k_mid_used += 1

        pnl_s = pd.Series(new_pnls)
        st    = calc_stats(pnl_s)
        trigger_rate = n_triggered / len(new_pnls) * 100
        total_trades = len(new_pnls)

        results.append({
            'k_low':    k_low,
            'k_mid':    k_mid,
            'k_high':   k_high,
            'pct_win':  pct_win,
            'sharpe':   round(st['sharpe'], 3),
            'total':    round(st['total'], 0),
            'max_dd':   round(st['max_dd'], 1),
            'wr':       round(st['wr'], 1),
            'rr':       round(st['rr'], 2),
            'trig%':    round(trigger_rate, 1),
            'low%':     round(k_low_used  / total_trades * 100, 1),
            'mid%':     round(k_mid_used  / total_trades * 100, 1),
            'high%':    round(k_high_used / total_trades * 100, 1),
            'sharpe_diff': round(st['sharpe'] - baseline['sharpe'], 3),
        })

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"结果汇总（按 Sharpe 排序，基准={baseline['sharpe']:.3f}）")
    print(f"{'─'*90}")

    best  = results_df.iloc[0]
    worst = results_df.iloc[-1]

    print(f"最好：Sharpe {best['sharpe']:.3f} ({best['sharpe_diff']:+.3f}) "
          f"| k=({best['k_low']},{best['k_mid']},{best['k_high']}) pct_win={int(best['pct_win'])} "
          f"| 触发率 {best['trig%']:.1f}%")
    print(f"最差：Sharpe {worst['sharpe']:.3f} ({worst['sharpe_diff']:+.3f}) "
          f"| k=({worst['k_low']},{worst['k_mid']},{worst['k_high']}) pct_win={int(worst['pct_win'])} "
          f"| 触发率 {worst['trig%']:.1f}%")

    n_better = (results_df['sharpe_diff'] > 0).sum()
    print(f"\n优于基准的组合：{n_better}/{len(results_df)}")

    print(f"\n全部结果（按 Sharpe 排序）:")
    header = (f"{'k_low':>6} {'k_mid':>5} {'k_high':>6} {'win':>4} "
              f"{'sharpe':>7} {'diff':>6} {'total':>9} {'maxDD%':>7} "
              f"{'WR%':>5} {'RR':>5} {'trig%':>6} {'low%':>5} {'mid%':>5} {'high%':>6}")
    print(header)
    print('─' * 95)
    for _, r in results_df.iterrows():
        sign = '+' if r['sharpe_diff'] >= 0 else ''
        print(f"{r['k_low']:>6.1f} {r['k_mid']:>5.1f} {r['k_high']:>6.1f} {int(r['pct_win']):>4} "
              f"{r['sharpe']:>7.3f} {sign}{r['sharpe_diff']:>5.3f} "
              f"{r['total']:>+9,.0f} {r['max_dd']:>7.1f} "
              f"{r['wr']:>5.1f} {r['rr']:>5.2f} {r['trig%']:>6.1f} "
              f"{r['low%']:>5.1f} {r['mid%']:>5.1f} {r['high%']:>6.1f}")

    # ── 详细标的视图（最优参数）──────────────────────────────────────────────
    if args.detail:
        best_row = results_df.iloc[0]
        bk_low  = best_row['k_low']
        bk_mid  = best_row['k_mid']
        bk_high = best_row['k_high']
        bwin    = int(best_row['pct_win'])

        print(f"\n{'─'*72}")
        print(f"最优参数按标的拆分（k_low={bk_low}, k_mid={bk_mid}, k_high={bk_high}, pct_win={bwin}）")
        print(f"{'─'*72}")

        for (ver, sym), (trades, ccxt_sym) in sorted(all_trades.items()):
            ohlcv = all_ohlcv[ccxt_sym]
            atr_s, hh_s, ll_s = base_cache[ccxt_sym]
            atr_pct_s = atr_pct_cache[(ccxt_sym, bwin)]

            # 原始基准（未加 CE）
            orig_pnls = list(trades['pnl'])

            # 自适应 CE 出场
            new_pnls, n_trig = [], 0
            for _, row in trades.iterrows():
                new_pnl, triggered, _, _ = simulate_trade_adaptive(
                    row, ohlcv, atr_s, hh_s, ll_s, atr_pct_s,
                    bk_low, bk_mid, bk_high
                )
                new_pnls.append(new_pnl)
                if triggered: n_trig += 1

            orig_s = calc_stats(pd.Series(orig_pnls), n_combos=1)
            new_s  = calc_stats(pd.Series(new_pnls), n_combos=1)
            diff   = new_s['sharpe'] - orig_s['sharpe']
            sign   = '+' if diff >= 0 else ''
            print(f"  {ver}_{sym:<5} | Sharpe {orig_s['sharpe']:5.3f} → {new_s['sharpe']:5.3f} "
                  f"({sign}{diff:.3f}) | "
                  f"Total {orig_s['total']:+8,.0f} → {new_s['total']:+8,.0f} | "
                  f"触发率 {n_trig/len(trades)*100:.1f}%")

    print(f"\n总交易笔数：{baseline['n']}")
    print(f"参数总组合：{len(results_df)}")
    print(f"\n结论参考：若优于基准的组合 < 20%，ATR 自适应 k 方向可视为无效。")

if __name__ == '__main__':
    run()
