"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
Chandelier Exit 出场优化验证（2026-06-03）

验证在持仓中途使用 Chandelier Exit（ATR 追踪止损）提前出场，
是否能改善趋势跟踪策略表现。

核心逻辑：
  - 入场信号：沿用原始 TV EMA 策略信号（从 xlsx 读取）
  - 出场触发：
      多头：close < highest_high(period) - k * ATR(atr_period)
      空头：close > lowest_low(period)  + k * ATR(atr_period)
  - 选项A（替换出场）：触发时立即出场，不再等原信号出场
  - 选项B（提前出场）：触发时出场，等下一笔原始信号重新计算
    （与 ADX 验证的选项A等效，即等下一个 EMA 信号）

时间框架：8H（Binance 永续，479m 策略最接近时间框架）
标的：BTC/ETH/SOL/DOGE，v1 + v2，共 8 个策略×标的组合
数据：TV 导出 xlsx（2026-05-22）+ Binance 8H OHLCV（2019-2026）

时区：TV 导出 UTC+8 naive → 减 8 小时 → UTC，与 Binance 对齐

参数网格：
  - ce_period（最高最低价回望）：10, 14, 20, 30
  - atr_period（ATR 周期）：10, 14, 20
  - k（ATR 乘数）：1.5, 2.0, 2.5, 3.0, 3.5

总组合数：4 × 3 × 5 = 60 组

用法：
  python chandelier_exit_validation.py
  python chandelier_exit_validation.py --detail
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--detail', action='store_true', help='按标的展示详细结果')
args = parser.parse_args()

BASE_CAPITAL = 10_000   # 固定仓位（与历史对比基准一致）
COMMISSION   = 0.0008   # 万分之八（taker 含滑点），双边

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
    # TV UTC+8 → UTC
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

# ─── Chandelier Exit 指标预计算 ───────────────────────────────────────────────

def calc_chandelier(ohlcv_df, ce_period, atr_period, k):
    """
    预计算全局 Chandelier Exit 止损线。
    返回 (long_stop, short_stop) 两个 Series，index = dt。

    long_stop  = highest_high(ce_period) - k * ATR(atr_period)   # 多头追踪止损
    short_stop = lowest_low(ce_period)   + k * ATR(atr_period)   # 空头追踪止损
    """
    high  = ohlcv_df['high']
    low   = ohlcv_df['low']
    close = ohlcv_df['close']

    # ATR（Wilder 平滑，与 ADX 脚本一致）
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    # 最高/最低价回望
    hh = high.rolling(ce_period, min_periods=1).max()
    ll = low.rolling(ce_period,  min_periods=1).min()

    long_stop  = hh - k * atr
    short_stop = ll + k * atr

    return long_stop, short_stop

# ─── 单笔交易模拟 ─────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price):
    """按方向和价格重算 PnL（含双边手续费）"""
    if entry_price <= 0 or exit_price <= 0: return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * BASE_CAPITAL
    cost  = BASE_CAPITAL * COMMISSION * 2
    return gross - cost

def simulate_trade(trade, ohlcv_df, long_stop_s, short_stop_s):
    """
    在 [entry_dt, exit_dt) 窗口内扫描：
      多头：close < long_stop  → 触发提前出场
      空头：close > short_stop → 触发提前出场
    首个触发 bar 的收盘价作为提前出场价。
    若无触发，返回原始 PnL（即 CE 不起作用）。

    返回：(new_pnl, triggered: bool, early_dt or None)
    """
    entry_dt  = trade['entry_dt']
    exit_dt   = trade['exit_dt']
    direction = trade['direction']

    # 持仓期间的 bar：entry_dt 之后（不含 entry bar 本身），exit_dt 之前
    mask = (ohlcv_df.index > entry_dt) & (ohlcv_df.index < exit_dt)
    window = ohlcv_df[mask]
    if window.empty:
        return trade['pnl'], False, None

    stop_s = long_stop_s if direction == 1 else short_stop_s

    for dt, row in window.iterrows():
        stop_val = stop_s.loc[dt] if dt in stop_s.index else np.nan
        if np.isnan(stop_val): continue

        triggered = (direction == 1 and row['close'] < stop_val) or \
                    (direction == -1 and row['close'] > stop_val)

        if triggered:
            new_pnl = recalc_pnl(direction, trade['entry_price'], row['close'])
            if np.isnan(new_pnl):
                return trade['pnl'], False, None
            return new_pnl, True, dt

    return trade['pnl'], False, None

# ─── 汇总统计 ─────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years=7, n_combos=8):
    """n_combos 用于回撤分母"""
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

# 参数网格
CE_PERIODS  = [10, 14, 20, 30]
ATR_PERIODS = [10, 14, 20]
K_MULTS     = [1.5, 2.0, 2.5, 3.0, 3.5]

def run():
    print("=== Chandelier Exit 出场优化验证 ===")
    print(f"时间框架：8H（Binance 永续，对应 479m 策略）")
    print(f"多头止损：highest_high(period) - k × ATR(atr_period)")
    print(f"空头止损：lowest_low(period)  + k × ATR(atr_period)")
    print(f"参数网格：ce_period={CE_PERIODS}, atr_period={ATR_PERIODS}, k={K_MULTS}")
    n_combos = len(CE_PERIODS) * len(ATR_PERIODS) * len(K_MULTS)
    print(f"总组合数：{n_combos}\n")

    # ── 加载所有数据 ──────────────────────────────────────────────────────────
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

    loaded = [(ver, sym) for ver, sym in all_trades]
    print(f"已加载：{[f'{v}_{s}' for v,s in loaded]}")

    print("\n拉取 OHLCV 数据...")
    all_ohlcv = {}
    for (ver, sym), (trades, ccxt_sym) in all_trades.items():
        if ccxt_sym not in all_ohlcv:
            all_ohlcv[ccxt_sym] = fetch_ohlcv(ccxt_sym)

    # ── 基准统计 ──────────────────────────────────────────────────────────────
    baseline_pnl = pd.concat([t for t, _ in all_trades.values()])['pnl']
    baseline = calc_stats(baseline_pnl)
    print(f"\n{'─'*72}")
    print(f"基准（原始出场）：Sharpe={baseline['sharpe']:.3f}  "
          f"Total={baseline['total']:+,.0f}  MaxDD={baseline['max_dd']:.1f}%  "
          f"WR={baseline['wr']:.1f}%  RR={baseline['rr']:.2f}  N={baseline['n']}")
    print(f"{'─'*72}")

    # ── 预计算所有参数组合下的 CE 止损线 ─────────────────────────────────────
    # 按 (symbol, ce_period, atr_period, k) 缓存，避免重复计算
    ce_cache = {}
    print(f"\n预计算 CE 指标（{len(all_ohlcv)} 个标的 × {n_combos} 组参数）...", end=' ', flush=True)
    for ccxt_sym, ohlcv in all_ohlcv.items():
        for ce_p, atr_p, k in product(CE_PERIODS, ATR_PERIODS, K_MULTS):
            key = (ccxt_sym, ce_p, atr_p, k)
            ce_cache[key] = calc_chandelier(ohlcv, ce_p, atr_p, k)
    print("done")

    # ── 参数扫描 ──────────────────────────────────────────────────────────────
    results = []
    param_combos = list(product(CE_PERIODS, ATR_PERIODS, K_MULTS))

    print(f"\n扫描 {len(param_combos)} 组参数...")
    for ce_p, atr_p, k in param_combos:
        new_pnls = []
        n_triggered = 0

        for (ver, sym), (trades, ccxt_sym) in all_trades.items():
            ohlcv = all_ohlcv[ccxt_sym]
            long_stop, short_stop = ce_cache[(ccxt_sym, ce_p, atr_p, k)]
            for _, row in trades.iterrows():
                new_pnl, triggered, _ = simulate_trade(row, ohlcv, long_stop, short_stop)
                new_pnls.append(new_pnl)
                if triggered: n_triggered += 1

        pnl_s = pd.Series(new_pnls)
        st    = calc_stats(pnl_s)
        trigger_rate = n_triggered / len(new_pnls) * 100

        results.append({
            'ce_period':  ce_p,
            'atr_period': atr_p,
            'k':          k,
            'sharpe':     round(st['sharpe'], 3),
            'total':      round(st['total'], 0),
            'max_dd':     round(st['max_dd'], 1),
            'wr':         round(st['wr'], 1),
            'rr':         round(st['rr'], 2),
            'triggered%': round(trigger_rate, 1),
            'sharpe_diff': round(st['sharpe'] - baseline['sharpe'], 3),
        })

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"结果汇总（按 Sharpe 排序，基准={baseline['sharpe']:.3f}）")
    print(f"{'─'*80}")

    best  = results_df.iloc[0]
    worst = results_df.iloc[-1]

    print(f"最好：Sharpe {best['sharpe']:.3f} ({best['sharpe_diff']:+.3f}) "
          f"| ce={int(best['ce_period'])}, atr={int(best['atr_period'])}, k={best['k']:.1f} "
          f"| 触发率 {best['triggered%']:.1f}%")
    print(f"最差：Sharpe {worst['sharpe']:.3f} ({worst['sharpe_diff']:+.3f}) "
          f"| ce={int(worst['ce_period'])}, atr={int(worst['atr_period'])}, k={worst['k']:.1f} "
          f"| 触发率 {worst['triggered%']:.1f}%")

    n_better = (results_df['sharpe_diff'] > 0).sum()
    print(f"\n优于基准的组合：{n_better}/{len(results_df)}")

    print(f"\nTop 15（按 Sharpe）:")
    header = f"{'ce_p':>5} {'atr_p':>5} {'k':>4} {'sharpe':>7} {'diff':>6} {'total':>9} {'maxDD%':>7} {'WR%':>5} {'RR':>5} {'trig%':>6}"
    print(header)
    print('─' * 70)
    for _, r in results_df.head(15).iterrows():
        sign = '+' if r['sharpe_diff'] >= 0 else ''
        print(f"{int(r['ce_period']):>5} {int(r['atr_period']):>5} {r['k']:>4.1f} "
              f"{r['sharpe']:>7.3f} {sign}{r['sharpe_diff']:>5.3f} "
              f"{r['total']:>+9,.0f} {r['max_dd']:>7.1f} "
              f"{r['wr']:>5.1f} {r['rr']:>5.2f} {r['triggered%']:>6.1f}")

    # k 维度分析（关键：k 越小止损越紧，应该越容易改善/恶化）
    print(f"\n按 k 分组的平均 Sharpe（基准={baseline['sharpe']:.3f}）:")
    for k_val in K_MULTS:
        sub = results_df[results_df['k'] == k_val]
        avg_sharpe = sub['sharpe'].mean()
        avg_trig   = sub['triggered%'].mean()
        diff_sign  = '+' if avg_sharpe > baseline['sharpe'] else ''
        print(f"  k={k_val:.1f}: avg Sharpe {avg_sharpe:.3f} ({diff_sign}{avg_sharpe - baseline['sharpe']:.3f}) "
              f"| avg 触发率 {avg_trig:.1f}%")

    # ── 详细标的视图（最优参数） ──────────────────────────────────────────────
    if args.detail:
        best_row = results_df.iloc[0]
        best_ce_p  = int(best_row['ce_period'])
        best_atr_p = int(best_row['atr_period'])
        best_k     = best_row['k']

        print(f"\n{'─'*72}")
        print(f"最优参数按标的拆分（ce={best_ce_p}, atr={best_atr_p}, k={best_k:.1f}）")
        print(f"{'─'*72}")

        for (ver, sym), (trades, ccxt_sym) in sorted(all_trades.items()):
            ohlcv = all_ohlcv[ccxt_sym]
            long_stop, short_stop = ce_cache[(ccxt_sym, best_ce_p, best_atr_p, best_k)]
            new_pnls, n_trig = [], 0
            for _, row in trades.iterrows():
                new_pnl, triggered, _ = simulate_trade(row, ohlcv, long_stop, short_stop)
                new_pnls.append(new_pnl)
                if triggered: n_trig += 1

            orig_s = calc_stats(trades['pnl'], n_combos=1)
            new_s  = calc_stats(pd.Series(new_pnls), n_combos=1)
            diff   = new_s['sharpe'] - orig_s['sharpe']
            sign   = '+' if diff >= 0 else ''
            print(f"  {ver}_{sym:<5} | Sharpe {orig_s['sharpe']:5.3f} → {new_s['sharpe']:5.3f} "
                  f"({sign}{diff:.3f}) | "
                  f"Total {orig_s['total']:+8,.0f} → {new_s['total']:+8,.0f} | "
                  f"触发率 {n_trig/len(trades)*100:.1f}%")

    print(f"\n总交易笔数（基准）：{baseline['n']}")
    print(f"参数总组合：{len(results_df)}")
    print(f"\n结论参考：若优于基准的组合 < 5%，CE 出场方向可视为无效。")

if __name__ == '__main__':
    run()
