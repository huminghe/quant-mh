"""
Donchian 通道突破入场过滤验证（2026-05-29）

核心假设：
  EMA 信号给方向，Donchian 通道突破给确认。
  做多信号：入场价 > Donchian 上轨（最近 N 根 bar 最高价）
  做空信号：入场价 < Donchian 下轨（最近 N 根 bar 最低价）

变体：
  - N：10 / 20 / 55（经典 Turtle 参数）
  - 时间框架：1h / 2h / 4h（与策略信号框架对齐）
  - 宽松版：价格 > Donchian 上轨 × (1 - slack)，允许略低于上轨也算突破

时区修正：TV 时间 -8h 转 UTC

用法：
  python /tmp/donchian_filter.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

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

CAPITAL  = 20000
TEST_TFS = ['1h', '2h', '4h']
DON_NS   = [10, 20, 55]

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，含方向和入场价格，TV UTC+8 -8h 转 UTC"""
    path = downloads / fname
    if not path.exists():
        print(f'  [WARN] 文件不存在: {fname}')
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
            pnl  = row[col_idx['净损益 USDT']]
            px   = row[col_idx['价格 USDT']]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ:
                by_num[num]['entry_dt']  = pd.Timestamp(dt)
                by_num[num]['entry_px']  = float(px) if px is not None else np.nan
                by_num[num]['direction'] = 'long' if '多头' in typ else 'short'
            elif '出场' in typ and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values()
                if 'entry_dt' in d and 'pnl' in d and 'direction' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf):
    key = (symbol, tf)
    if key in ohlcv_cache: return ohlcv_cache[key]
    print(f'  拉取 {symbol} {tf}...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not bars: break
        all_bars.extend(bars)
        if len(bars) < 1000: break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[ns]')
    df = df.set_index('dt').sort_index()
    ohlcv_cache[key] = df
    print('done')
    return df

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def calc_donchian(df, n):
    """返回 (上轨, 下轨)，用 shift(1) 避免用当前 bar 数据"""
    upper = df['high'].rolling(n).max().shift(1)
    lower = df['low'].rolling(n).min().shift(1)
    return upper, lower

# ─── 过滤器评估 ───────────────────────────────────────────────────────────────

def eval_donchian(trades_df, ohlcv_df, n, slack=0.0):
    """
    Donchian 突破过滤：
      做多：entry_px > upper × (1 - slack)
      做空：entry_px < lower × (1 + slack)
    slack=0 为严格突破，slack>0 为宽松版（允许略低于上轨）
    """
    upper, lower = calc_donchian(ohlcv_df, n)

    don_df = pd.DataFrame({'ts': ohlcv_df.index, 'upper': upper.values, 'lower': lower.values})
    don_df['ts'] = don_df['ts'].astype('datetime64[ns]')

    trades = trades_df.copy()
    trades['entry_dt'] = trades['entry_dt'].astype('datetime64[ns]')

    merged = pd.merge_asof(
        trades.sort_values('entry_dt'),
        don_df.sort_values('ts'),
        left_on='entry_dt', right_on='ts',
        direction='backward'
    )

    # 判断是否突破
    is_long  = merged['direction'] == 'long'
    is_short = merged['direction'] == 'short'
    px = merged['entry_px']

    long_break  = is_long  & (px > merged['upper']  * (1 - slack))
    short_break = is_short & (px < merged['lower']  * (1 + slack))
    nan_mask    = merged['upper'].isna() | merged['lower'].isna()

    mask = nan_mask | long_break | short_break

    base = trades['pnl'].sum() * (CAPITAL / 50000)
    filt = merged[mask]['pnl'].sum() * (CAPITAL / 50000)
    pct  = (filt - base) / abs(base) * 100 if base != 0 else 0
    keep = mask.sum() / len(trades) * 100

    # 多空拆分
    long_base  = merged[is_long]['pnl'].sum()  * (CAPITAL / 50000)
    short_base = merged[is_short]['pnl'].sum() * (CAPITAL / 50000)
    long_filt  = merged[mask & is_long]['pnl'].sum()  * (CAPITAL / 50000)
    short_filt = merged[mask & is_short]['pnl'].sum() * (CAPITAL / 50000)

    return {
        'pct': pct, 'keep': keep,
        'long_pct':  (long_filt  - long_base)  / abs(long_base)  * 100 if long_base  != 0 else 0,
        'short_pct': (short_filt - short_base) / abs(short_base) * 100 if short_base != 0 else 0,
    }

# ─── 主验证 ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("Donchian 通道突破入场过滤验证（时区修正版）")
    print("=" * 72)

    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df

    print(f"\n已加载 {len(all_trades)} 个版本×标的\n")

    for tf in TEST_TFS:
        print(f"\n{'='*60}")
        print(f"时间框架：{tf}")
        print(f"{'='*60}")

        # 预拉取 K 线
        ohlcv_map = {}
        for (ver, sym, ccxt_sym) in all_trades:
            if ccxt_sym not in ohlcv_map:
                ohlcv_map[ccxt_sym] = fetch_ohlcv(ccxt_sym, tf)

        print(f"\n  {'N':>4}  {'slack':>6}  {'变化':>8}  {'保留':>6}  {'多头':>8}  {'空头':>8}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}")

        results = []
        for n in DON_NS:
            for slack in [0.0, 0.005, 0.01]:
                # 合并所有版本×标的
                total_base = total_filt = 0
                long_base = long_filt = short_base = short_filt = 0

                for (ver, sym, ccxt_sym), trades_df in all_trades.items():
                    ohlcv = ohlcv_map[ccxt_sym]
                    r = eval_donchian(trades_df, ohlcv, n, slack)
                    b = trades_df['pnl'].sum() * (CAPITAL / 50000)
                    total_base += b
                    total_filt += b * (1 + r['pct'] / 100)

                    is_long  = trades_df['direction'] == 'long'
                    is_short = trades_df['direction'] == 'short'
                    lb = trades_df[is_long]['pnl'].sum()  * (CAPITAL / 50000)
                    sb = trades_df[is_short]['pnl'].sum() * (CAPITAL / 50000)
                    long_base  += lb
                    short_base += sb
                    long_filt  += lb * (1 + r['long_pct']  / 100)
                    short_filt += sb * (1 + r['short_pct'] / 100)

                pct   = (total_filt - total_base) / abs(total_base) * 100
                lpct  = (long_filt  - long_base)  / abs(long_base)  * 100 if long_base  != 0 else 0
                spct  = (short_filt - short_base) / abs(short_base) * 100 if short_base != 0 else 0

                # keep rate：用第一个标的估算（各标的相近）
                sample_key = list(all_trades.keys())[0]
                sample_trades = all_trades[sample_key]
                sample_ohlcv  = ohlcv_map[sample_key[2]]
                keep = eval_donchian(sample_trades, sample_ohlcv, n, slack)['keep']

                results.append({'n': n, 'slack': slack, 'pct': pct, 'keep': keep})
                slack_str = f'{slack*100:.1f}%'
                print(f"  {n:>4}  {slack_str:>6}  {pct:>+7.1f}%  {keep:>5.0f}%  {lpct:>+7.1f}%  {spct:>+7.1f}%")

        valid = [r for r in results if not np.isnan(r['pct'])]
        if valid:
            best  = max(valid, key=lambda x: x['pct'])
            worst = min(valid, key=lambda x: x['pct'])
            print(f"\n  最好：N={best['n']} slack={best['slack']*100:.1f}%  {best['pct']:+.1f}%（保留{best['keep']:.0f}%）")
            print(f"  最差：N={worst['n']} slack={worst['slack']*100:.1f}%  {worst['pct']:+.1f}%（保留{worst['keep']:.0f}%）")

    print("\n" + "=" * 72)
    print("完成")

if __name__ == '__main__':
    run()
