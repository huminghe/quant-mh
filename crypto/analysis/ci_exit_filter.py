"""
CI 出场过滤验证（2026-05-27）

模拟：持仓期间如果 2H CI 超过阈值，在该 K 线收盘提前出场。
对比原始 PnL vs 提前出场后的 PnL，扫描多个阈值（50~80）。

出场价格估算：用触发 K 线的收盘价，乘以原始仓位大小反推 PnL。
注意：这是近似模拟，实际 Pine Script 实现会有细微差异。

用法：
  python ci_exit_filter.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')
files = {
    'BTC': ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH': ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL': ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE':('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}
CAPITAL = 20000
CI_N = 10          # 与入场过滤保持一致
TF   = '2h'        # 占位，实际在主流程中循环多个时间框架

# ─── 加载交易数据（含出场时间和出场价格）────────────────────────────────────────

def load_trades(fname):
    path = downloads / fname
    if not path.exists():
        print(f"  文件不存在: {fname}")
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {v: i for i, v in enumerate(rows[0]) if v}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num  = row[col_idx['交易 #']]
            typ  = str(row[col_idx['类型']])
            dt   = row[col_idx['日期和时间']]
            px   = row[col_idx['价格 USDT']]
            sz   = row[col_idx['大小（数量）']]   # 合约数量
            pnl  = row[col_idx['净损益 USDT']]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
                by_num[num]['entry_px'] = float(px) if px else np.nan
                by_num[num]['size']     = float(sz) if sz else np.nan
                # 判断方向：多头进场=做多，空头进场=做空
                by_num[num]['side'] = 'long' if '多头' in typ else 'short'
            elif '出场' in typ and pnl is not None:
                by_num[num]['exit_dt']  = pd.Timestamp(dt)
                by_num[num]['exit_px']  = float(px) if px else np.nan
                by_num[num]['pnl']      = float(pnl)
        except:
            continue
    wb.close()
    needed = ['entry_dt', 'exit_dt', 'entry_px', 'exit_px', 'size', 'side', 'pnl']
    rows_out = [d for d in by_num.values() if all(k in d for k in needed)]
    df = pd.DataFrame(rows_out)
    for col in ['entry_dt', 'exit_dt']:
        df[col] = df[col].dt.tz_localize(None).astype('datetime64[us]')
    return df.sort_values('entry_dt').reset_index(drop=True)

# ─── 拉取 OHLCV 并计算 CI ────────────────────────────────────────────────────

ohlcv_cache = {}

def fetch_ohlcv(symbol, tf=TF):
    key = (symbol, tf)
    if key in ohlcv_cache: return ohlcv_cache[key]
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601('2019-01-01T00:00:00Z')
    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch: break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break
    df = pd.DataFrame(all_ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[us]')
    df = df.set_index('dt').sort_index()
    ohlcv_cache[key] = df
    return df

def compute_ci(df, n=CI_N):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    hl_range = (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(n)
    return ci

# ─── 核心：模拟提前出场 ──────────────────────────────────────────────────────

def simulate_early_exit(trades, ohlcv, ci_series, threshold):
    """
    对每笔交易，检查持仓期间是否有 CI > threshold 的 K 线。
    如果有，用第一根触发 K 线的收盘价计算提前出场的 PnL。
    返回每笔交易的模拟 PnL（未触发则保留原始 PnL）。
    """
    results = []
    for _, trade in trades.iterrows():
        entry_dt  = trade['entry_dt']
        exit_dt   = trade['exit_dt']
        entry_px  = trade['entry_px']
        size      = trade['size']
        side      = trade['side']
        orig_pnl  = trade['pnl']

        # 持仓期间的 K 线（不含入场 K 线，含出场 K 线）
        mask = (ohlcv.index > entry_dt) & (ohlcv.index <= exit_dt)
        bars_in_trade = ohlcv[mask].copy()
        ci_in_trade   = ci_series[mask]

        if bars_in_trade.empty:
            results.append({'orig_pnl': orig_pnl, 'sim_pnl': orig_pnl, 'early_exit': False})
            continue

        # 找第一根 CI > threshold 的 K 线
        trigger = ci_in_trade[ci_in_trade > threshold]
        if trigger.empty:
            results.append({'orig_pnl': orig_pnl, 'sim_pnl': orig_pnl, 'early_exit': False})
            continue

        # 用触发 K 线的收盘价估算 PnL
        trigger_dt = trigger.index[0]
        exit_px_sim = bars_in_trade.loc[trigger_dt, 'close']

        if side == 'long':
            sim_pnl = (exit_px_sim - entry_px) * size
        else:
            sim_pnl = (entry_px - exit_px_sim) * size

        # 扣除手续费（万8双边，出场一侧万4）
        fee = exit_px_sim * size * 0.0004
        sim_pnl -= fee

        results.append({'orig_pnl': orig_pnl, 'sim_pnl': sim_pnl, 'early_exit': True})

    return pd.DataFrame(results)

# ─── 主流程 ──────────────────────────────────────────────────────────────────

thresholds  = [50, 55, 58, 61.8, 65, 68, 70, 75, 80]
timeframes  = ['2h', '4h', '8h', '1d']   # 1d ≈ 24h

print("加载交易数据...")
all_trades = {}
for sym, (fname, ccxt_sym) in files.items():
    t = load_trades(fname)
    if not t.empty:
        all_trades[sym] = (t, ccxt_sym)
        print(f"  {sym}: {len(t)} 笔交易")

print("\n拉取 OHLCV 数据（多时间框架）...")
for sym, (trades, ccxt_sym) in all_trades.items():
    for tf in timeframes:
        fetch_ohlcv(ccxt_sym, tf)
    print(f"  {sym} 完成")

# ─── 逐时间框架结果 ──────────────────────────────────────────────────────────

for tf in timeframes:
    print("\n" + "="*90)
    print(f"  CI({CI_N}) {tf} 出场过滤效果  （固定资本 {CAPITAL:,} USDT/标的）")
    print(f"  出场逻辑：持仓期间第一根 CI > 阈值的 K 线收盘价出场")
    print("="*90)

    w = [6, 10] + [14]*len(thresholds)
    header = ['标的', '基准PnL'] + [f'CI>{t}' for t in thresholds]
    print('  ' + '  '.join(h.center(wi) for h, wi in zip(header, w)))
    print('  ' + '  '.join('-'*wi for wi in w))

    summary_orig = np.zeros(len(thresholds))
    summary_sim  = np.zeros(len(thresholds))
    summary_base = 0.0
    per_sym_detail = {}

    for sym, (trades, ccxt_sym) in all_trades.items():
        ohlcv = fetch_ohlcv(ccxt_sym, tf)
        ci    = compute_ci(ohlcv)

        base_pnl = trades['pnl'].sum()
        summary_base += base_pnl

        row_vals   = [sym, f'{base_pnl:+,.0f}']
        detail_rows = []

        for i, thresh in enumerate(thresholds):
            res       = simulate_early_exit(trades, ohlcv, ci, thresh)
            sim_total = res['sim_pnl'].sum()
            n_early   = res['early_exit'].sum()
            n_total   = len(res)
            diff      = sim_total - base_pnl
            summary_sim[i] += sim_total
            marker = '★' if diff > 3000 else ('✗' if diff < -3000 else ' ')
            row_vals.append(f'{sim_total:+,.0f}({diff:+,.0f}){marker}')
            detail_rows.append((thresh, sim_total, diff, n_early, n_total))

        per_sym_detail[sym] = detail_rows
        print('  ' + '  '.join(str(v).center(wi) for v, wi in zip(row_vals, w)))

    # 合计行
    print('  ' + '  '.join('-'*wi for wi in w))
    total_row = ['合计', f'{summary_base:+,.0f}']
    for i, thresh in enumerate(thresholds):
        diff   = summary_sim[i] - summary_base
        marker = '★' if diff > 8000 else ('✗' if diff < -8000 else ' ')
        total_row.append(f'{summary_sim[i]:+,.0f}({diff:+,.0f}){marker}')
    print('  ' + '  '.join(str(v).center(wi) for v, wi in zip(total_row, w)))

    # 触发率
    print(f"\n  触发率（提前出场笔数 / 总笔数）")
    w2     = [6] + [12]*len(thresholds)
    header2 = ['标的'] + [f'CI>{t}' for t in thresholds]
    print('  ' + '  '.join(h.center(wi) for h, wi in zip(header2, w2)))
    print('  ' + '  '.join('-'*wi for wi in w2))
    for sym, detail_rows in per_sym_detail.items():
        row = [sym]
        for thresh, sim_total, diff, n_early, n_total in detail_rows:
            pct = n_early / n_total * 100 if n_total > 0 else 0
            row.append(f'{n_early}/{n_total}({pct:.0f}%)')
        print('  ' + '  '.join(str(v).center(wi) for v, wi in zip(row, w2)))

print("\n完成。")
