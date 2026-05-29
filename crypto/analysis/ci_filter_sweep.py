"""
所有指标的过滤效果验证：看总收益（固定资本 20000 USDT/标的）
用 4H 时间级别，与 CI 保持一致
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')
files = {
    'BTC_ema':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH_ema':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL_ema':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE_ema': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}
CAPITAL = 20000

def load_trades(fname):
    path = downloads / fname
    if not path.exists(): return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    col_idx = {v: i for i, v in enumerate(header) if v}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx['交易 #']]; typ = str(row[col_idx['类型']])
            dt = row[col_idx['日期和时间']]; pnl = row[col_idx['净损益 USDT']]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ: by_num[num]['entry_dt'] = pd.Timestamp(dt)
            elif '出场' in typ and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = df['entry_dt'].dt.tz_localize(None).astype('datetime64[us]')
    return df

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

def compute_all_indicators(df):
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    ret = c.pct_change()

    # Choppiness(14)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum14 = tr.rolling(14).sum()
    ci = 100 * np.log10(atr_sum14 / (h.rolling(14).max() - l.rolling(14).min()).replace(0,np.nan)) / np.log10(14)

    # ADX(14)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0.0)
    ndm = dn.where((dn>up)&(dn>0), 0.0)
    pdi = 100*pdm.ewm(alpha=1/14,adjust=False).mean()/atr14
    ndi = 100*ndm.ewm(alpha=1/14,adjust=False).mean()/atr14
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100/(1 + gain/loss.replace(0,np.nan))

    # ATR百分位（252根4H = 约42天）
    atr_pct = atr14.rolling(252).rank(pct=True) * 100

    # BBW 布林带宽
    ma20 = c.rolling(20).mean()
    bbw = (c.rolling(20).std() * 4) / ma20 * 100

    # MACD柱
    macd_hist = (c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean())
    macd_hist = macd_hist - macd_hist.ewm(span=9,adjust=False).mean()

    # 相对MA200（4H下200根=约33天）
    ma200 = c.rolling(200).mean()
    rel_ma200 = (c - ma200) / ma200 * 100

    # 21根动量（4H×21=3.5天）
    mom21 = c.pct_change(21) * 100

    # 52周价格位置（4H×252=42天，用252根近似）
    price_pct = c.rolling(252).rank(pct=True) * 100

    # 成交量比率
    vol_ratio = v / v.rolling(20).mean()

    return pd.DataFrame({
        'ci': ci, 'adx': adx, 'rsi': rsi, 'atr_pct': atr_pct,
        'bbw': bbw, 'macd_hist': macd_hist, 'rel_ma200': rel_ma200,
        'mom21': mom21, 'price_pct': price_pct, 'vol_ratio': vol_ratio,
    }, index=df.index)

def merge_indicators(trades, symbol):
    df = fetch_ohlcv(symbol, '4h')
    ind = compute_all_indicators(df).reset_index()
    ind['dt'] = ind['dt'].astype('datetime64[us]')
    return pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind.rename(columns={'dt': 'entry_dt'}),
        on='entry_dt', direction='backward'
    )

print("加载数据...")
all_trades_raw = {}
for strat, (fname, symbol) in files.items():
    t = load_trades(fname)
    if not t.empty: all_trades_raw[strat] = (t, symbol)

print("拉取4H数据...")
for strat, (trades, symbol) in all_trades_raw.items():
    fetch_ohlcv(symbol, '4h')

print("合并指标...")
all_merged = []
for strat, (trades, symbol) in all_trades_raw.items():
    m = merge_indicators(trades, symbol)
    m['strat'] = strat
    all_merged.append(m)
combined = pd.concat(all_merged, ignore_index=True)
base_total = combined['pnl'].sum()
base_ret = base_total / (CAPITAL*4) * 100
print(f"基准：{len(combined)} 笔，总PnL {base_total:+,.0f} USDT ({base_ret:+.1f}%)\n")

# ─── 各指标阈值扫描 ───────────────────────────────────────────────────────────
# 格式：(指标名, 显示名, [(阈值, 方向, 说明)])
# 方向 'le'=保留<=阈值, 'ge'=保留>=阈值
indicator_configs = [
    ('ci',        'Choppiness(14)',   [(38.2,'le','强趋势'),(50,'le',''),(55,'le',''),(61.8,'le','标准阈值'),(65,'le',''),(70,'le','')]),
    ('adx',       'ADX(14)',          [(15,'ge','强趋势'),(20,'ge',''),(25,'ge',''),(30,'ge',''),(35,'ge','极强趋势')]),
    ('rsi',       'RSI(14)',          [(30,'le','超卖'),(40,'le',''),(60,'ge',''),(70,'ge','超买'),(40,'ge','排除超卖'),(60,'le','排除超买')]),
    ('atr_pct',   'ATR百分位',         [(25,'ge','低波动过滤'),(40,'ge',''),(50,'ge',''),(60,'ge',''),(75,'ge','高波动')]),
    ('bbw',       'BBW布林带宽',        [(5,'ge','极窄过滤'),(8,'ge',''),(10,'ge',''),(15,'ge',''),(20,'ge','宽带')]),
    ('rel_ma200', '相对MA200%',        [(-5,'ge','MA200以上'),(0,'ge',''),(5,'ge',''),(10,'ge',''),(0,'le','MA200以下')]),
    ('mom21',     '21根动量%',         [(-5,'ge','负动量过滤'),(-2,'ge',''),(0,'ge','正动量'),(5,'ge',''),(0,'le','负动量')]),
    ('price_pct', '价格位置(252根)',    [(20,'ge','非底部'),(40,'ge',''),(60,'ge',''),(80,'ge','高位'),(20,'le','底部')]),
    ('vol_ratio', '成交量比率',         [(0.5,'ge','非极低量'),(0.8,'ge',''),(1.2,'le','非放量'),(2.0,'ge','放量')]),
]

print("="*85)
print("  各指标过滤效果（4H级别，固定资本 20000 USDT/标的，合计 80000 USDT）")
print("  格式：总PnL (收益率%) [vs基准差值]")
print("="*85)

for ind_col, ind_name, thresholds in indicator_configs:
    t = combined.dropna(subset=[ind_col])
    n_valid = len(t)
    base = t['pnl'].sum()
    base_r = base / (CAPITAL*4) * 100
    print(f"\n  [{ind_name}]  有效笔数:{n_valid}  基准:{base:+,.0f}({base_r:+.1f}%)")
    print(f"  {'条件':<28} {'保留笔数(%)':>12} {'总PnL':>12} {'收益率':>9} {'vs基准':>10}")
    print("  " + "-"*75)
    for thresh, direction, note in thresholds:
        if direction == 'le':
            mask = t[ind_col] <= thresh
            label = f"≤{thresh} {note}"
        else:
            mask = t[ind_col] >= thresh
            label = f"≥{thresh} {note}"
        kept = t[mask]
        if len(kept) < 50: continue
        total = kept['pnl'].sum()
        ret = total / (CAPITAL*4) * 100
        pct = len(kept)/n_valid*100
        diff = total - base
        marker = " ★" if diff > 5000 else (" ✗" if diff < -5000 else "")
        print(f"  {label:<28} {len(kept):>6}({pct:>4.0f}%) {total:>+11,.0f} {ret:>+8.1f}% {diff:>+9,.0f}{marker}")

# ─── 组合过滤：CI + 最优其他指标 ─────────────────────────────────────────────
print("\n" + "="*85)
print("  组合过滤：CI≤61.8 + 其他指标（4H，合计80000 USDT）")
print("="*85)
t = combined.dropna(subset=['ci','adx','rsi','atr_pct','bbw','rel_ma200','mom21'])
base = t['pnl'].sum()
base_r = base / (CAPITAL*4) * 100
print(f"\n  基准（无过滤）：{len(t)} 笔  {base:+,.0f} ({base_r:+.1f}%)")
print(f"  {'条件':<40} {'保留笔数(%)':>12} {'总PnL':>12} {'收益率':>9} {'vs基准':>10}")
print("  " + "-"*80)

combos = [
    ("CI≤61.8",                          (t['ci']<=61.8)),
    ("CI≤61.8 + ADX≥20",                 (t['ci']<=61.8)&(t['adx']>=20)),
    ("CI≤61.8 + ADX≥25",                 (t['ci']<=61.8)&(t['adx']>=25)),
    ("CI≤61.8 + ATR%≥40",                (t['ci']<=61.8)&(t['atr_pct']>=40)),
    ("CI≤61.8 + ATR%≥50",                (t['ci']<=61.8)&(t['atr_pct']>=50)),
    ("CI≤61.8 + RSI≥40",                 (t['ci']<=61.8)&(t['rsi']>=40)),
    ("CI≤61.8 + RSI≥40 + RSI≤70",        (t['ci']<=61.8)&(t['rsi']>=40)&(t['rsi']<=70)),
    ("CI≤61.8 + MA200以上",               (t['ci']<=61.8)&(t['rel_ma200']>=0)),
    ("CI≤61.8 + 动量≥0",                  (t['ci']<=61.8)&(t['mom21']>=0)),
    ("CI≤61.8 + ADX≥20 + ATR%≥40",       (t['ci']<=61.8)&(t['adx']>=20)&(t['atr_pct']>=40)),
    ("ADX≥20 + ATR%≥40",                 (t['adx']>=20)&(t['atr_pct']>=40)),
    ("ADX≥20 + CI≤61.8 + 动量≥0",        (t['adx']>=20)&(t['ci']<=61.8)&(t['mom21']>=0)),
]

for label, mask in combos:
    kept = t[mask]
    if len(kept) < 50: continue
    total = kept['pnl'].sum()
    ret = total / (CAPITAL*4) * 100
    pct = len(kept)/len(t)*100
    diff = total - base
    marker = " ★" if diff > 8000 else (" ✗" if diff < -5000 else "")
    print(f"  {label:<40} {len(kept):>6}({pct:>4.0f}%) {total:>+11,.0f} {ret:>+8.1f}% {diff:>+9,.0f}{marker}")

print("\n完成。")
