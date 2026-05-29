"""
CI(10) 2H ≤50 入场过滤效果验证
- 按标的分别展示：基准PnL、过滤后PnL、胜率对比、年度分解
- 支持 v1/v2/v3_205m/v3_3h 等多个策略版本
- 用法：修改下方 files 字典，指定要分析的策略文件
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')

# 修改这里切换策略版本
# v1
files = {
    'BTC': ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH': ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL': ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE':('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}
CI_N = 10
CI_TF = '2h'
CI_THRESH = 50

def load_trades(fname):
    path = downloads / fname
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

def compute_ci(df, n=10):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    hl = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(atr_sum / hl) / np.log10(n)

total_base = total_keep = 0

for sym, (fname, symbol) in files.items():
    trades = load_trades(fname)
    df = fetch_ohlcv(symbol, CI_TF)
    ci = compute_ci(df, CI_N).reset_index()
    ci.columns = ['entry_dt', 'ci']
    ci['entry_dt'] = ci['entry_dt'].astype('datetime64[us]')
    merged = pd.merge_asof(trades.sort_values('entry_dt'), ci, on='entry_dt', direction='backward')

    total = len(merged)
    base_pnl = merged['pnl'].sum()
    keep = merged[merged['ci'].isna() | (merged['ci'] <= CI_THRESH)]
    filtered = merged[merged['ci'].notna() & (merged['ci'] > CI_THRESH)]
    keep_pnl = keep['pnl'].sum()
    filt_pnl = filtered['pnl'].sum()
    keep_wr = (keep['pnl'] > 0).mean() * 100 if len(keep) > 0 else 0
    filt_wr = (filtered['pnl'] > 0).mean() * 100 if len(filtered) > 0 else 0
    ci_vals = merged['ci'].dropna()

    total_base += base_pnl
    total_keep += keep_pnl

    print(f'\n=== {sym} ({symbol}) ===')
    print(f'总交易笔数：{total}')
    print(f'基准总PnL：{base_pnl:+,.0f} USDT  胜率：{(merged["pnl"]>0).mean()*100:.1f}%')
    print(f'CI均值：{ci_vals.mean():.1f}  中位数：{ci_vals.median():.1f}  >50占比：{(ci_vals>CI_THRESH).mean()*100:.1f}%')
    print(f'')
    print(f'  保留（CI≤{CI_THRESH}）：{len(keep)}笔（{len(keep)/total*100:.1f}%）  PnL：{keep_pnl:+,.0f}  胜率：{keep_wr:.1f}%')
    print(f'  过滤（CI>{CI_THRESH}）：{len(filtered)}笔（{len(filtered)/total*100:.1f}%）  PnL：{filt_pnl:+,.0f}  胜率：{filt_wr:.1f}%')
    print(f'  过滤效果：{keep_pnl - base_pnl:+,.0f} USDT')

    merged['year'] = merged['entry_dt'].dt.year
    keep['year'] = keep['entry_dt'].dt.year
    print(f'\n  年度分解：')
    print(f'  {"年份":>6}  {"基准PnL":>10}  {"过滤后PnL":>10}  {"差值":>8}  {"过滤笔数":>8}')
    for yr in sorted(merged['year'].unique()):
        b = merged[merged['year']==yr]['pnl'].sum()
        k = keep[keep['year']==yr]['pnl'].sum() if yr in keep['year'].values else 0
        f_n = len(filtered[filtered['entry_dt'].dt.year==yr]) if len(filtered)>0 else 0
        print(f'  {yr:>6}  {b:>+10,.0f}  {k:>+10,.0f}  {k-b:>+8,.0f}  {f_n:>8}')

print(f'\n{"="*50}')
print(f'合计  基准：{total_base:+,.0f}  过滤后：{total_keep:+,.0f}  提升：{total_keep-total_base:+,.0f}')
