"""
ADX 出场优化验证（2026-06-03）

验证在持仓中途检测到 ADX 弱化时提前出场，是否能改善策略表现。

核心逻辑：
  - 入场信号：沿用原始 TV EMA 策略信号（从 xlsx 读取）
  - 出场触发（两者结合，AND 逻辑）：
      静态：ADX < threshold（绝对值偏弱）
      动态：ADX 从近 N 根峰值下降 > drop_pct%（趋势衰减）
  - 同时满足才提前出场

选项A（等信号）：提前出场后，下一笔原始策略信号才重新入场
选项B（等ADX恢复）：提前出场后，ADX > 恢复阈值时立即重新入场，
                    以当时K线收盘价计算剩余盈亏，拼回原笔交易的损益

时间框架：8H（Binance 永续，479m 策略最近似时间框架）
标的：BTC/ETH/SOL/DOGE，v1 + v2，共 8 个策略×标的组合
数据：TV 导出 xlsx（2026-05-22）+ Binance 8H OHLCV（2019-2026）

时区：TV 导出 UTC+8 naive → 减 8 小时 → UTC，与 Binance 对齐

用法：
  python adx_exit_validation.py
  python adx_exit_validation.py --detail   # 按标的展示详细结果
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

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

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
            pnl_key = '净损益 USDT' if '净损益 USDT' in col_idx else 'Profit USDT'
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

# ─── ADX 计算（Wilder 平滑，周期=14）────────────────────────────────────────

def calc_adx(ohlcv_df, period=14):
    """返回带 dt index 的 adx Series"""
    high, low, close = ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1/period, adjust=False).mean()
    up    = high.diff()
    down  = -low.diff()
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_m  = down.where((down > up) & (down > 0), 0.0)
    di_p  = 100 * dm_p.ewm(alpha=1/period, adjust=False).mean() / atr
    di_m  = 100 * dm_m.ewm(alpha=1/period, adjust=False).mean() / atr
    dx    = (100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10))
    adx   = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx  # index = dt

# ─── 出场逻辑 ─────────────────────────────────────────────────────────────────

def find_early_exit(ohlcv_df, adx_s, entry_dt, exit_dt,
                    static_thresh, drop_pct, lookback_n):
    """
    在 [entry_dt, exit_dt] 窗口内，找第一个同时满足以下条件的 8H bar：
      1. adx < static_thresh（静态阈值）
      2. adx < peak_adx * (1 - drop_pct)，peak_adx = 过去 lookback_n 根的最大值（动态下降）
    返回触发 bar 的 dt，若无触发返回 None。
    """
    mask = (adx_s.index >= entry_dt) & (adx_s.index < exit_dt)
    window = adx_s[mask]
    if window.empty: return None

    rolling_peak = adx_s.rolling(lookback_n, min_periods=1).max()

    for dt, adx_val in window.items():
        peak = rolling_peak.loc[dt]
        static_ok  = adx_val < static_thresh
        dynamic_ok = adx_val < peak * (1 - drop_pct)
        if static_ok and dynamic_ok:
            return dt
    return None

def find_adx_recovery(adx_s, from_dt, recovery_thresh, exit_dt):
    """
    从 from_dt 之后找第一个 adx >= recovery_thresh 的 bar。
    recovery_thresh = static_thresh（进场时标准一致）。
    若超过 exit_dt 仍未恢复，返回 None（表示本笔交易已结束）。
    """
    mask = (adx_s.index > from_dt) & (adx_s.index <= exit_dt)
    window = adx_s[mask]
    recovered = window[window >= recovery_thresh]
    if recovered.empty: return None
    return recovered.index[0]

# ─── PnL 重算 ─────────────────────────────────────────────────────────────────

def recalc_pnl(direction, entry_price, exit_price, capital=BASE_CAPITAL):
    """
    按方向和价格重算 PnL（含双边手续费）。
    direction: +1=多, -1=空
    """
    if entry_price <= 0 or exit_price <= 0: return np.nan
    gross = direction * (exit_price - entry_price) / entry_price * capital
    cost  = capital * COMMISSION * 2
    return gross - cost

# ─── 单笔交易模拟 ─────────────────────────────────────────────────────────────

def simulate_trade_option_a(trade, ohlcv_df, adx_s, params):
    """
    选项A：提前出场，不再入场，直接跳过原出场到 ADX 出场之间的持仓。
    返回修改后的 pnl（若无提前触发则返回原 pnl）。
    """
    static_thresh = params['static_thresh']
    drop_pct      = params['drop_pct']
    lookback_n    = params['lookback_n']

    early_dt = find_early_exit(ohlcv_df, adx_s,
                               trade['entry_dt'], trade['exit_dt'],
                               static_thresh, drop_pct, lookback_n)
    if early_dt is None:
        return trade['pnl'], False  # 未触发，原 pnl

    # 用 early_dt bar 的收盘价作为提前出场价
    try:
        early_close = ohlcv_df.loc[early_dt, 'close']
    except KeyError:
        return trade['pnl'], False

    new_pnl = recalc_pnl(trade['direction'], trade['entry_price'], early_close)
    if np.isnan(new_pnl):
        return trade['pnl'], False
    return new_pnl, True

def simulate_trade_option_b(trade, ohlcv_df, adx_s, params):
    """
    选项B：提前出场后等 ADX 恢复再重新入场。
    - 提前出场 pnl = 入场价 → 提前出场价
    - 若 ADX 在原出场时间前恢复，重新入场：re_entry → 原出场价再算一段 pnl
    - 两段 pnl 相加（各扣手续费）
    若无提前触发，返回原 pnl。
    """
    static_thresh = params['static_thresh']
    drop_pct      = params['drop_pct']
    lookback_n    = params['lookback_n']
    recovery_thresh = static_thresh  # 恢复标准与触发标准一致

    early_dt = find_early_exit(ohlcv_df, adx_s,
                               trade['entry_dt'], trade['exit_dt'],
                               static_thresh, drop_pct, lookback_n)
    if early_dt is None:
        return trade['pnl'], False

    try:
        early_close = ohlcv_df.loc[early_dt, 'close']
    except KeyError:
        return trade['pnl'], False

    # 第一段：入场 → 提前出场
    pnl_1 = recalc_pnl(trade['direction'], trade['entry_price'], early_close)
    if np.isnan(pnl_1):
        return trade['pnl'], False

    # 找 ADX 恢复点
    re_entry_dt = find_adx_recovery(adx_s, early_dt, recovery_thresh, trade['exit_dt'])
    if re_entry_dt is None:
        # ADX 在原出场前未恢复，只有第一段
        return pnl_1, True

    try:
        re_entry_close = ohlcv_df.loc[re_entry_dt, 'close']
    except KeyError:
        return pnl_1, True

    # 第二段：重新入场 → 原出场价
    pnl_2 = recalc_pnl(trade['direction'], re_entry_close, trade['exit_price'])
    if np.isnan(pnl_2):
        return pnl_1, True

    return pnl_1 + pnl_2, True

# ─── 汇总统计 ─────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years=7, n_combos=8):
    """
    n_combos：标的×版本组合数，用于回撤分母（与 ATR 脚本一致）
    """
    total   = pnl_series.sum()
    wr      = (pnl_series > 0).mean() * 100
    wins    = pnl_series[pnl_series > 0]
    losses  = pnl_series[pnl_series < 0]
    rr      = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 0
    cum     = pnl_series.cumsum()
    dd      = ((cum - cum.cummax()) / (BASE_CAPITAL * n_combos) * 100).min()
    sharpe  = ((pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series) / n_years)
               if pnl_series.std() > 0 else 0)
    return dict(total=total, wr=wr, rr=rr, max_dd=dd, sharpe=sharpe, n=len(pnl_series))

# ─── 主流程 ───────────────────────────────────────────────────────────────────

# 参数网格
STATIC_THRESHOLDS = [15, 20, 25]
DROP_PCTS         = [0.10, 0.20, 0.30]
LOOKBACK_NS       = [3, 5, 10]

def run():
    print("=== ADX 出场优化验证 ===")
    print(f"时间框架：8H（Binance 永续，对应 479m 策略）")
    print(f"ADX 周期：14（Wilder 平滑）")
    print(f"出场条件：ADX < 静态阈值 AND ADX 从 N 根峰值下降 > X%（AND 逻辑）")
    print(f"参数网格：静态={STATIC_THRESHOLDS}, 下降%={DROP_PCTS}, 回望={LOOKBACK_NS}")
    print(f"总组合数：{len(STATIC_THRESHOLDS)*len(DROP_PCTS)*len(LOOKBACK_NS)*2} "
          f"（{len(STATIC_THRESHOLDS)*len(DROP_PCTS)*len(LOOKBACK_NS)} 组参数 × 2 选项）\n")

    # ── 加载所有数据 ──────────────────────────────────────────────────────────
    print("加载交易记录...")
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades[(ver, sym)] = (trades, ccxt_sym)

    print("\n拉取 OHLCV 数据...")
    all_ohlcv = {}
    all_adx   = {}
    for (ver, sym), (trades, ccxt_sym) in all_trades.items():
        if ccxt_sym not in all_ohlcv:
            ohlcv = fetch_ohlcv(ccxt_sym)
            all_ohlcv[ccxt_sym] = ohlcv
            all_adx[ccxt_sym]   = calc_adx(ohlcv)

    # ── 基准统计（无 ADX 出场）────────────────────────────────────────────────
    baseline_pnl = pd.concat([t for t, _ in all_trades.values()])['pnl']
    baseline = calc_stats(baseline_pnl)
    print(f"\n{'─'*72}")
    print(f"基准（无 ADX 出场）：Sharpe={baseline['sharpe']:.3f}  "
          f"Total={baseline['total']:+,.0f}  MaxDD={baseline['max_dd']:.1f}%  "
          f"N={baseline['n']}")
    print(f"{'─'*72}")

    # ── 参数扫描 ──────────────────────────────────────────────────────────────
    results = []
    param_combos = list(product(STATIC_THRESHOLDS, DROP_PCTS, LOOKBACK_NS))

    for static_thresh, drop_pct, lookback_n in param_combos:
        params = dict(static_thresh=static_thresh, drop_pct=drop_pct, lookback_n=lookback_n)

        for option, simulate_fn in [('A', simulate_trade_option_a), ('B', simulate_trade_option_b)]:
            new_pnls = []
            n_triggered = 0

            for (ver, sym), (trades, ccxt_sym) in all_trades.items():
                ohlcv = all_ohlcv[ccxt_sym]
                adx_s = all_adx[ccxt_sym]
                for _, row in trades.iterrows():
                    new_pnl, triggered = simulate_fn(row, ohlcv, adx_s, params)
                    new_pnls.append(new_pnl)
                    if triggered: n_triggered += 1

            pnl_s = pd.Series(new_pnls)
            st    = calc_stats(pnl_s)
            trigger_rate = n_triggered / len(new_pnls) * 100

            results.append({
                'option': option,
                'static': static_thresh,
                'drop%':  int(drop_pct * 100),
                'lookback': lookback_n,
                'sharpe':   round(st['sharpe'], 3),
                'total':    round(st['total'], 0),
                'max_dd':   round(st['max_dd'], 1),
                'wr':       round(st['wr'], 1),
                'rr':       round(st['rr'], 2),
                'triggered%': round(trigger_rate, 1),
            })

    results_df = pd.DataFrame(results)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    for option in ['A', 'B']:
        opt_df = results_df[results_df['option'] == option].copy()
        opt_df = opt_df.sort_values('sharpe', ascending=False)
        best   = opt_df.iloc[0]
        worst  = opt_df.iloc[-1]

        print(f"\n选项{option}（{'等 EMA 信号重新入场' if option=='A' else 'ADX 恢复后立即重新入场'}）")
        print(f"  基准 Sharpe {baseline['sharpe']:.3f} → "
              f"最好 {best['sharpe']:.3f}（static={best['static']}, "
              f"drop={best['drop%']}%, lookback={best['lookback']}）, "
              f"触发率 {best['triggered%']:.1f}%")
        print(f"  最差 {worst['sharpe']:.3f}（static={worst['static']}, "
              f"drop={worst['drop%']}%, lookback={worst['lookback']}）")

        # 前 10 行
        print(f"\n  Top 10（按 Sharpe 排序）:")
        header = f"  {'static':>6} {'drop%':>5} {'lookback':>8} {'sharpe':>7} {'total':>9} {'maxDD%':>7} {'trig%':>6}"
        print(header)
        print(f"  {'─'*55}")
        for _, r in opt_df.head(10).iterrows():
            print(f"  {r['static']:>6} {r['drop%']:>5} {r['lookback']:>8} "
                  f"{r['sharpe']:>7.3f} {r['total']:>+9,.0f} {r['max_dd']:>7.1f} {r['triggered%']:>6.1f}")

    # ── 详细标的视图 ──────────────────────────────────────────────────────────
    if args.detail:
        # 取选项A和选项B各自的最优参数，按标的拆分
        for option in ['A', 'B']:
            opt_df = results_df[results_df['option'] == option]
            best_row = opt_df.sort_values('sharpe', ascending=False).iloc[0]
            best_params = dict(
                static_thresh=int(best_row['static']),
                drop_pct=best_row['drop%'] / 100,
                lookback_n=int(best_row['lookback']),
            )
            sim_fn = simulate_trade_option_a if option == 'A' else simulate_trade_option_b

            print(f"\n{'─'*72}")
            print(f"选项{option} 最优参数详细拆分（static={best_params['static_thresh']}, "
                  f"drop={int(best_params['drop_pct']*100)}%, lookback={best_params['lookback_n']}）")
            print(f"{'─'*72}")

            for (ver, sym), (trades, ccxt_sym) in sorted(all_trades.items()):
                ohlcv = all_ohlcv[ccxt_sym]
                adx_s = all_adx[ccxt_sym]
                new_pnls, triggered_cnt = [], 0
                for _, row in trades.iterrows():
                    new_pnl, triggered = sim_fn(row, ohlcv, adx_s, best_params)
                    new_pnls.append(new_pnl)
                    if triggered: triggered_cnt += 1

                orig_s = calc_stats(trades['pnl'], n_combos=1)
                new_s  = calc_stats(pd.Series(new_pnls), n_combos=1)
                diff   = new_s['sharpe'] - orig_s['sharpe']
                sign   = '+' if diff >= 0 else ''
                print(f"  {ver}_{sym:<5} | Sharpe {orig_s['sharpe']:5.3f} → {new_s['sharpe']:5.3f} "
                      f"({sign}{diff:.3f}) | "
                      f"Total {orig_s['total']:+8,.0f} → {new_s['total']:+8,.0f} | "
                      f"触发率 {triggered_cnt/len(trades)*100:.1f}%")

    print(f"\n总交易笔数（基准）：{baseline['n']}")
    print(f"参数总组合：{len(results_df)} 组（{len(param_combos)} 参数 × 2 选项）")
    print(f"\n结论参考：若全部 Sharpe < 基准 {baseline['sharpe']:.3f}，ADX 出场方向无效。")

if __name__ == '__main__':
    run()
