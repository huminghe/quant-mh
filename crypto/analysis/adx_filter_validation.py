"""
ADX 过滤信号验证（2026-05-27）

验证 ADX 在4个维度（自身8H / BTC 8H / 自身日线 / BTC日线）对4个策略
（BTC/ETH/SOL/DOGE_ema）的入场过滤效果。

结论：16张表无一呈现单调趋势，ADX 对这批策略无预测价值。
BTC 8H ADX < 10 仅占全部交易1.2%，即使有效也无实用价值。

用法：
  python adx_filter_validation.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import ccxt
import openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')
files = {
    'BTC_ema':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH_ema':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL_ema':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE_ema': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}

def load_trades(fname):
    wb = openpyxl.load_workbook(downloads / fname, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {v: i for i, v in enumerate(rows[0]) if v}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        num = row[col_idx['交易 #']]
        typ = str(row[col_idx['类型']])
        dt  = row[col_idx['日期和时间']]
        pnl = row[col_idx['净损益 USDT']]
        if num is None or dt is None: continue
        if num not in by_num: by_num[num] = {}
        if '进场' in typ:
            by_num[num]['entry_dt'] = pd.Timestamp(dt)
        elif '出场' in typ and pnl is not None:
            by_num[num]['exit_dt'] = pd.Timestamp(dt)
            by_num[num]['pnl'] = float(pnl)
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'exit_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = df['entry_dt'].dt.tz_localize(None).astype('datetime64[ms]')
    return df

adx_cache = {}
def get_adx(symbol, timeframe):
    key = (symbol, timeframe)
    if key in adx_cache:
        return adx_cache[key]
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601('2019-01-01T00:00:00Z')
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms').astype('datetime64[ms]')
    df = df.set_index('dt').drop(columns=['timestamp'])
    df = df[~df.index.duplicated()].sort_index()
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up, down = high.diff(), -low.diff()
    dm_p = up.where((up>down)&(up>0), 0)
    dm_m = down.where((down>up)&(down>0), 0)
    di_p = 100 * dm_p.ewm(alpha=1/14, adjust=False).mean() / atr
    di_m = 100 * dm_m.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = (100 * (di_p - di_m).abs() / (di_p + di_m)).fillna(0)
    adx = dx.ewm(alpha=1/14, adjust=False).mean().reset_index()
    adx.columns = ['dt', 'adx']
    adx_cache[key] = adx
    return adx

def attach_adx(trades, symbol, timeframe):
    adx_df = get_adx(symbol, timeframe)
    t = trades.sort_values('entry_dt').rename(columns={'entry_dt': 'dt'})
    merged = pd.merge_asof(t, adx_df, on='dt', direction='backward')
    return merged.rename(columns={'dt': 'entry_dt'})

bins = [
    (0,   10,  '0~10（极弱）'),
    (10,  15,  '10~15（很弱）'),
    (15,  20,  '15~20（弱）'),
    (20,  25,  '20~25（中等）'),
    (25,  35,  '25~35（强）'),
    (35,  999, '35+（极强）'),
]

def print_table(df, title):
    w1, w2, w3, w4 = 16, 7, 7, 7
    sep   = f"├{'─'*(w1+2)}┼{'─'*(w2+2)}┼{'─'*(w3+2)}┼{'─'*(w4+2)}┤"
    top   = f"┌{'─'*(w1+2)}┬{'─'*(w2+2)}┬{'─'*(w3+2)}┬{'─'*(w4+2)}┐"
    bot   = f"└{'─'*(w1+2)}┴{'─'*(w2+2)}┴{'─'*(w3+2)}┴{'─'*(w4+2)}┘"
    hdr   = f"│ {'ADX区间':<{w1}} │ {'笔数':>{w2}} │ {'均PnL':>{w3}} │ {'胜率':>{w4}} │"

    print(f"\n  {title}")
    print(f"  {top}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for lo, hi, lbl in bins:
        mask = (df['adx'] >= lo) & (df['adx'] < hi)
        t = df[mask]
        if len(t) < 3:
            row = f"│ {lbl:<{w1}} │ {'—':>{w2}} │ {'—':>{w3}} │ {'—':>{w4}} │"
        else:
            wr  = f"{(t['pnl']>0).mean()*100:.1f}%"
            pnl = f"{t['pnl'].mean():+.1f}"
            row = f"│ {lbl:<{w1}} │ {len(t):>{w2}} │ {pnl:>{w3}} │ {wr:>{w4}} │"
        print(f"  {row}")
        if (lo, hi, lbl) != bins[-1]:
            print(f"  {sep}")
    print(f"  {bot}")

# 预拉数据
print("拉取数据中...")
for sym in ['BTC/USDT','ETH/USDT','SOL/USDT','DOGE/USDT']:
    for tf in ['8h','1d']:
        get_adx(sym, tf)

configs = [
    ('自身 8H ADX',   lambda sym: (sym,        '8h')),
    ('BTC  8H ADX',   lambda sym: ('BTC/USDT', '8h')),
    ('自身 日线 ADX',  lambda sym: (sym,        '1d')),
    ('BTC  日线 ADX',  lambda sym: ('BTC/USDT', '1d')),
]

for strat, (fname, sym) in files.items():
    trades = load_trades(fname)
    print(f"\n{'━'*60}")
    print(f"  {strat}（{sym}）  共 {len(trades)} 笔交易")
    print(f"{'━'*60}")
    for cfg_name, get_sym_tf in configs:
        adx_sym, tf = get_sym_tf(sym)
        merged = attach_adx(trades, adx_sym, tf)
        print_table(merged, cfg_name)

