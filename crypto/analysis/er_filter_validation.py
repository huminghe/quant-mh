"""
ER 过滤深度验证（2026-05-27）

1. 年度稳定性：ER≥0.3 vs CI≤50，逐年对比，看是否每年都正向
2. 逐标的一致性：4个标的分别展示，看是否跨标的稳定

用法：
  python er_filter_validation.py
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
ER_N    = 10
TF      = '2h'

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    path = downloads / fname
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
    df['year'] = df['entry_dt'].dt.year
    return df.sort_values('entry_dt').reset_index(drop=True)

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

def compute_indicators(df, n=ER_N):
    c = df['close']; h = df['high']; l = df['low']
    # ER
    direction  = c.diff(n).abs()
    volatility = c.diff().abs().rolling(n).sum()
    er = direction / volatility.replace(0, np.nan)
    # CI
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum  = tr.rolling(n).sum()
    hl_range = (h.rolling(n).max() - l.rolling(n).min()).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(n)
    return er.rename('er'), ci.rename('ci')

def merge_ind(trades, symbol):
    ohlcv = fetch_ohlcv(symbol)
    er, ci = compute_indicators(ohlcv)
    ind = pd.concat([er, ci], axis=1).reset_index()
    ind['dt'] = ind['dt'].astype('datetime64[us]')
    return pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind.rename(columns={'dt': 'entry_dt'}),
        on='entry_dt', direction='backward'
    )

# ─── 加载数据 ─────────────────────────────────────────────────────────────────

print("加载数据...")
all_merged = {}
for sym, (fname, ccxt_sym) in files.items():
    t = load_trades(fname)
    m = merge_ind(t, ccxt_sym)
    all_merged[sym] = m.dropna(subset=['er', 'ci'])
    print(f"  {sym}: {len(m)} 笔")

combined = pd.concat(all_merged.values(), ignore_index=True)

# ─── 第一部分：年度稳定性 ─────────────────────────────────────────────────────

print("\n" + "="*100)
print(f"  年度稳定性验证：ER≥0.3 vs CI≤50  （2H，合计4标的，固定资本 {CAPITAL:,} USDT/标的）")
print("="*100)

years = sorted(combined['year'].unique())
conditions = {
    '无过滤':    lambda df: pd.Series([True]*len(df), index=df.index),
    'CI≤50':    lambda df: df['ci'] <= 50,
    'ER≥0.3':   lambda df: df['er'] >= 0.3,
    'CI+ER≥0.3':lambda df: (df['ci'] <= 50) & (df['er'] >= 0.3),
}

# 表头
col_w = 12
print(f"\n  {'年份':<6}", end='')
for cname in conditions:
    print(f"  {cname:>{col_w}}", end='')
print()
print(f"  {'':6}", end='')
for cname in conditions:
    print(f"  {'PnL(vs基准)':>{col_w}}", end='')
print()
print("  " + "-"*70)

year_totals = {c: [] for c in conditions}

for year in years:
    ydf = combined[combined['year'] == year]
    if len(ydf) < 10: continue
    print(f"  {year:<6}", end='')
    base_pnl = None
    for cname, cond_fn in conditions.items():
        mask = cond_fn(ydf)
        pnl  = ydf[mask]['pnl'].sum()
        n    = mask.sum()
        if cname == '无过滤':
            base_pnl = pnl
            print(f"  {pnl:>+{col_w},.0f}", end='')
        else:
            diff = pnl - base_pnl
            marker = '+' if diff > 0 else '-'
            print(f"  {pnl:>+{col_w-5},.0f}({diff:>+5,.0f})", end='')
        year_totals[cname].append(pnl)
    print()

# 合计行
print("  " + "-"*70)
print(f"  {'合计':<6}", end='')
base_total = None
for cname in conditions:
    total = sum(year_totals[cname])
    if cname == '无过滤':
        base_total = total
        print(f"  {total:>+{col_w},.0f}", end='')
    else:
        diff = total - base_total
        print(f"  {total:>+{col_w-5},.0f}({diff:>+5,.0f})", end='')
print()

# 正向年份统计
print(f"\n  正向年份数（vs无过滤）：", end='')
base_by_year = {}
for year in years:
    ydf = combined[combined['year'] == year]
    if len(ydf) < 10: continue
    base_by_year[year] = ydf['pnl'].sum()

for cname, cond_fn in conditions.items():
    if cname == '无过滤': continue
    pos = 0; total_years = 0
    for year in years:
        ydf = combined[combined['year'] == year]
        if len(ydf) < 10: continue
        total_years += 1
        pnl = ydf[cond_fn(ydf)]['pnl'].sum()
        if pnl > base_by_year[year]: pos += 1
    print(f"  {cname} {pos}/{total_years}", end='')
print()

# ─── 第二部分：逐标的年度明细 ────────────────────────────────────────────────

print("\n" + "="*100)
print("  逐标的年度明细：ER≥0.3 (2H)")
print("="*100)

for sym, df in all_merged.items():
    print(f"\n  ── {sym} ──")
    print(f"  {'年份':<6} {'笔数':>5} {'基准PnL':>10} {'ER≥0.3':>10} {'差值':>8} {'CI≤50':>10} {'差值':>8} {'胜率(基准)':>10} {'胜率(ER)':>9}")
    print("  " + "-"*80)

    sym_base = 0; sym_er = 0; sym_ci = 0
    for year in years:
        ydf = df[df['year'] == year]
        if len(ydf) < 5: continue
        base = ydf['pnl'].sum()
        er30 = ydf[ydf['er'] >= 0.3]['pnl'].sum()
        ci50 = ydf[ydf['ci'] <= 50]['pnl'].sum()
        wr_base = (ydf['pnl'] > 0).mean() * 100
        er_rows = ydf[ydf['er'] >= 0.3]
        wr_er   = (er_rows['pnl'] > 0).mean() * 100 if len(er_rows) > 0 else 0
        n_er    = (ydf['er'] >= 0.3).sum()
        sym_base += base; sym_er += er30; sym_ci += ci50
        diff_er = er30 - base; diff_ci = ci50 - base
        m_er = '★' if diff_er > 1000 else ('✗' if diff_er < -1000 else ' ')
        print(f"  {year:<6} {len(ydf):>5} {base:>+10,.0f} {er30:>+9,.0f}{m_er} {diff_er:>+7,.0f} {ci50:>+9,.0f}  {diff_ci:>+7,.0f} {wr_base:>9.1f}% {wr_er:>8.1f}%({n_er}笔)")

    print("  " + "-"*80)
    diff_er = sym_er - sym_base; diff_ci = sym_ci - sym_base
    print(f"  {'合计':<6} {len(df):>5} {sym_base:>+10,.0f} {sym_er:>+9,.0f}  {diff_er:>+7,.0f} {sym_ci:>+9,.0f}  {diff_ci:>+7,.0f}")

# ─── 第三部分：ER 分布分析 ────────────────────────────────────────────────────

print("\n" + "="*100)
print("  ER 值分布：入场时 ER 的分位数（了解阈值覆盖范围）")
print("="*100)

print(f"\n  {'标的':<6} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6}  {'ER<0.2':>8} {'ER 0.2-0.3':>11} {'ER 0.3-0.4':>11} {'ER>0.4':>8}")
print("  " + "-"*75)
for sym, df in all_merged.items():
    er = df['er'].dropna()
    p = np.percentile(er, [10, 25, 50, 75, 90])
    b1 = (er < 0.2).mean()*100
    b2 = ((er >= 0.2) & (er < 0.3)).mean()*100
    b3 = ((er >= 0.3) & (er < 0.4)).mean()*100
    b4 = (er >= 0.4).mean()*100
    print(f"  {sym:<6} {p[0]:>6.2f} {p[1]:>6.2f} {p[2]:>6.2f} {p[3]:>6.2f} {p[4]:>6.2f}  {b1:>7.1f}% {b2:>10.1f}% {b3:>10.1f}% {b4:>7.1f}%")

print("\n完成。")
