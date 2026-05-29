"""
Efficiency Ratio (ER) 入场过滤验证（2026-05-27）

ER = |Close - Close[N]| / Σ|Close[i] - Close[i-1]|
- ER 接近 1：价格走直线（强趋势）
- ER 接近 0：价格反复震荡（横盘）

验证：
1. ER 单独作为入场过滤（多个阈值 + 多个时间框架）
2. 与 CI(10) 2H ≤50 对比
3. CI + ER 组合是否优于单独使用

用法：
  python er_filter_sweep.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')
files = {
    'BTC':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}
CAPITAL = 20000
ER_N    = 10   # 与 CI 保持一致，方便对比

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    path = downloads / fname
    if not path.exists(): return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {v: i for i, v in enumerate(rows[0]) if v}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx['交易 #']]; typ = str(row[col_idx['类型']])
            dt  = row[col_idx['日期和时间']]; pnl = row[col_idx['净损益 USDT']]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
            elif '出场' in typ and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = df['entry_dt'].dt.tz_localize(None).astype('datetime64[us]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf):
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

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def compute_er(df, n=ER_N):
    """Efficiency Ratio：净位移 / 总路程，范围 0~1，越高越趋势"""
    c = df['close']
    direction  = c.diff(n).abs()
    volatility = c.diff().abs().rolling(n).sum()
    return direction / volatility.replace(0, np.nan)

def compute_ci(df, n=10):
    """Choppiness Index：越低越趋势，越高越震荡"""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum  = tr.rolling(n).sum()
    hl_range = (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    return 100 * np.log10(atr_sum / hl_range) / np.log10(n)

def merge_indicators(trades, symbol, tf):
    ohlcv = fetch_ohlcv(symbol, tf)
    er = compute_er(ohlcv).rename('er')
    ci = compute_ci(ohlcv).rename('ci')
    ind = pd.concat([er, ci], axis=1).reset_index()
    ind['dt'] = ind['dt'].astype('datetime64[us]')
    return pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind.rename(columns={'dt': 'entry_dt'}),
        on='entry_dt', direction='backward'
    )

# ─── 过滤效果计算 ─────────────────────────────────────────────────────────────

def filter_stats(df, mask, base_pnl, label, n_total):
    kept = df[mask]
    if len(kept) < 30:
        return None
    total = kept['pnl'].sum()
    diff  = total - base_pnl
    pct   = len(kept) / n_total * 100
    wr    = (kept['pnl'] > 0).mean() * 100
    marker = '★' if diff > 5000 else ('✗' if diff < -5000 else ' ')
    return dict(label=label, n=len(kept), pct=pct, total=total, diff=diff, wr=wr, marker=marker)

# ─── 主流程 ──────────────────────────────────────────────────────────────────

print("加载交易数据...")
all_trades = {}
for sym, (fname, ccxt_sym) in files.items():
    t = load_trades(fname)
    if not t.empty:
        all_trades[sym] = (t, ccxt_sym)
        print(f"  {sym}: {len(t)} 笔")

timeframes = ['1h', '2h', '4h']

print("\n拉取 OHLCV 数据...")
for sym, (_, ccxt_sym) in all_trades.items():
    for tf in timeframes:
        fetch_ohlcv(ccxt_sym, tf)
    print(f"  {sym} 完成")

# ─── 第一部分：ER 阈值扫描（多时间框架）────────────────────────────────────────

er_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

print("\n" + "="*95)
print(f"  ER({ER_N}) 入场过滤效果扫描  （固定资本 {CAPITAL:,} USDT/标的，合计 {CAPITAL*4:,} USDT）")
print(f"  过滤逻辑：入场时 ER ≥ 阈值才允许开仓（ER 越高越趋势）")
print("="*95)

for tf in timeframes:
    print(f"\n  ── 时间框架：{tf} ──")

    # 合并所有标的
    merged_list = []
    for sym, (trades, ccxt_sym) in all_trades.items():
        m = merge_indicators(trades, ccxt_sym, tf)
        m['sym'] = sym
        merged_list.append(m)
    combined = pd.concat(merged_list, ignore_index=True)
    t = combined.dropna(subset=['er', 'ci'])
    base_pnl = t['pnl'].sum()
    base_ret = base_pnl / (CAPITAL * 4) * 100
    n_total  = len(t)

    print(f"  基准：{n_total} 笔  总PnL {base_pnl:+,.0f} ({base_ret:+.1f}%)")
    print(f"  {'条件':<20} {'保留笔数(%)':>12} {'总PnL':>12} {'收益率':>9} {'vs基准':>10} {'胜率':>7}")
    print("  " + "-"*75)

    rows = []
    for thresh in er_thresholds:
        mask = t['er'] >= thresh
        s = filter_stats(t, mask, base_pnl, f'ER≥{thresh}', n_total)
        if s:
            rows.append(s)
            print(f"  ER≥{thresh:<16} {s['n']:>6}({s['pct']:>4.0f}%) {s['total']:>+11,.0f} {s['total']/(CAPITAL*4)*100:>+8.1f}% {s['diff']:>+9,.0f} {s['wr']:>6.1f}%{s['marker']}")

    # 同框架下 CI 基准（方便对比）
    ci_mask = t['ci'] <= 50
    s_ci = filter_stats(t, ci_mask, base_pnl, 'CI≤50(基准)', n_total)
    if s_ci:
        print(f"  {'CI≤50 [对照]':<20} {s_ci['n']:>6}({s_ci['pct']:>4.0f}%) {s_ci['total']:>+11,.0f} {s_ci['total']/(CAPITAL*4)*100:>+8.1f}% {s_ci['diff']:>+9,.0f} {s_ci['wr']:>6.1f}%{s_ci['marker']}")

# ─── 第二部分：CI + ER 组合 ───────────────────────────────────────────────────

print("\n" + "="*95)
print("  CI(10) 2H ≤50  +  ER 组合过滤效果")
print("  基础：CI≤50 已过滤，在此基础上再加 ER 条件")
print("="*95)

# 用 2h 时间框架
merged_list = []
for sym, (trades, ccxt_sym) in all_trades.items():
    m = merge_indicators(trades, ccxt_sym, '2h')
    m['sym'] = sym
    merged_list.append(m)
combined_2h = pd.concat(merged_list, ignore_index=True)
t2 = combined_2h.dropna(subset=['er', 'ci'])
base_pnl = t2['pnl'].sum()
n_total  = len(t2)

ci_only_mask = t2['ci'] <= 50
ci_only_pnl  = t2[ci_only_mask]['pnl'].sum()
ci_only_n    = ci_only_mask.sum()

print(f"\n  基准（无过滤）：{n_total} 笔  {base_pnl:+,.0f}")
print(f"  CI≤50 单独：   {ci_only_n} 笔({ci_only_n/n_total*100:.0f}%)  {ci_only_pnl:+,.0f}  vs基准 {ci_only_pnl-base_pnl:+,.0f}")
print(f"\n  {'条件':<30} {'保留笔数(%)':>12} {'总PnL':>12} {'vs无过滤':>10} {'vs CI单独':>11} {'胜率':>7}")
print("  " + "-"*85)

combos = [
    ('CI≤50 + ER≥0.15', (t2['ci'] <= 50) & (t2['er'] >= 0.15)),
    ('CI≤50 + ER≥0.2',  (t2['ci'] <= 50) & (t2['er'] >= 0.2)),
    ('CI≤50 + ER≥0.25', (t2['ci'] <= 50) & (t2['er'] >= 0.25)),
    ('CI≤50 + ER≥0.3',  (t2['ci'] <= 50) & (t2['er'] >= 0.3)),
    ('CI≤50 + ER≥0.35', (t2['ci'] <= 50) & (t2['er'] >= 0.35)),
    ('CI≤50 + ER≥0.4',  (t2['ci'] <= 50) & (t2['er'] >= 0.4)),
    ('ER≥0.2 单独',      t2['er'] >= 0.2),
    ('ER≥0.25 单独',     t2['er'] >= 0.25),
    ('ER≥0.3 单独',      t2['er'] >= 0.3),
]

for label, mask in combos:
    kept  = t2[mask]
    if len(kept) < 30: continue
    total = kept['pnl'].sum()
    wr    = (kept['pnl'] > 0).mean() * 100
    diff_base = total - base_pnl
    diff_ci   = total - ci_only_pnl
    pct = len(kept) / n_total * 100
    marker = '★' if diff_base > 8000 else ('✗' if diff_base < -5000 else ' ')
    print(f"  {label:<30} {len(kept):>6}({pct:>4.0f}%) {total:>+11,.0f} {diff_base:>+9,.0f} {diff_ci:>+10,.0f} {wr:>6.1f}%{marker}")

# ─── 第三部分：逐标的明细（最优配置）────────────────────────────────────────────

print("\n" + "="*95)
print("  逐标的明细：CI≤50 vs ER≥0.25(2h) vs CI≤50+ER≥0.25")
print("="*95)

print(f"\n  {'标的':<6} {'基准PnL':>10} {'CI≤50':>12} {'ER≥0.25':>12} {'CI+ER≥0.25':>14} {'胜率(基准)':>10} {'胜率(CI+ER)':>12}")
print("  " + "-"*80)

for sym, (trades, ccxt_sym) in all_trades.items():
    m = merge_indicators(trades, ccxt_sym, '2h').dropna(subset=['er', 'ci'])
    base  = m['pnl'].sum()
    ci50  = m[m['ci'] <= 50]['pnl'].sum()
    er25  = m[m['er'] >= 0.25]['pnl'].sum()
    combo = m[(m['ci'] <= 50) & (m['er'] >= 0.25)]['pnl'].sum()
    wr_base  = (m['pnl'] > 0).mean() * 100
    wr_combo = (m[(m['ci'] <= 50) & (m['er'] >= 0.25)]['pnl'] > 0).mean() * 100
    n_combo  = ((m['ci'] <= 50) & (m['er'] >= 0.25)).sum()
    print(f"  {sym:<6} {base:>+10,.0f} {ci50:>+11,.0f}({ci50-base:>+6,.0f}) {er25:>+11,.0f}({er25-base:>+6,.0f}) {combo:>+11,.0f}({combo-base:>+7,.0f}) {wr_base:>9.1f}% {wr_combo:>10.1f}%({n_combo}笔)")

print("\n完成。")
