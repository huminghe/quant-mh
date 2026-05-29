"""
CI 参数扫描：N(7/10/14/20) × 阈值(45/50/55) × 时间框架(30m/1h/2h)
支持多个策略版本，输出每个标的的最优配置
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')

# 修改这里切换策略版本
files = {
    'BTC': ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH': ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL': ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE':('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}

TFS = ['30m', '1h', '2h']
NS = [7, 10, 14, 20]
THRESHOLDS = [45, 50, 55]

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

def compute_ci(df, n):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    hl = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(atr_sum / hl) / np.log10(n)

# 预拉数据
print('拉取行情数据...')
symbols_needed = set(v[1] for v in files.values())
for symbol in symbols_needed:
    for tf in TFS:
        fetch_ohlcv(symbol, tf)
print('完成\n')

for sym, (fname, symbol) in files.items():
    trades = load_trades(fname)
    total = len(trades)
    base_pnl = trades['pnl'].sum()
    print(f'\n--- {sym} ({symbol})  总笔数:{total}  基准PnL:{base_pnl:+,.0f}  胜率:{(trades["pnl"]>0).mean()*100:.1f}% ---')
    print(f'  {"配置":<20} {"保留笔数(%)":>12} {"过滤后PnL":>12} {"vs基准":>10} {"保留胜率":>8} {"过滤胜率":>8}')
    print('  ' + '-'*75)

    best_diff = -999999
    best_cfg = ''

    for tf in TFS:
        df = fetch_ohlcv(symbol, tf)
        for n in NS:
            ci_series = compute_ci(df, n).reset_index()
            ci_series.columns = ['entry_dt', 'ci']
            ci_series['entry_dt'] = ci_series['entry_dt'].astype('datetime64[us]')
            merged = pd.merge_asof(trades.sort_values('entry_dt'), ci_series, on='entry_dt', direction='backward')

            for thresh in THRESHOLDS:
                keep = merged[merged['ci'].isna() | (merged['ci'] <= thresh)]
                filt = merged[merged['ci'].notna() & (merged['ci'] > thresh)]
                keep_pnl = keep['pnl'].sum()
                keep_wr = (keep['pnl'] > 0).mean() * 100 if len(keep) > 0 else 0
                filt_wr = (filt['pnl'] > 0).mean() * 100 if len(filt) > 0 else 0
                diff = keep_pnl - base_pnl
                pct = len(keep) / total * 100
                cfg = f'CI({n}) {tf} ≤{thresh}'
                if diff > best_diff:
                    best_diff = diff
                    best_cfg = cfg
                print(f'  {cfg:<20} {len(keep):>6}({pct:>4.0f}%) {keep_pnl:>+12,.0f} {diff:>+10,.0f} {keep_wr:>7.1f}% {filt_wr:>7.1f}%')
        print()

    print(f'  >> 最优：{best_cfg}  vs基准 {best_diff:+,.0f}')
