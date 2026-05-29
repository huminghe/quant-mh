"""
ER 过滤器 v1/v2 参数细化对比（2026-05-29）

对比：
1. N=7 vs N=10（v1/v2 最优 N 选择）
2. 阈值 0.30 vs 0.35（常用阈值对比）
3. [1]（前一根K线）vs 不加[1]（当前K线）的效果差异

用法：
  python er_v1v2_param_detail.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')

V1_FILES = {
    'BTC':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}
V2_FILES = {
    'BTC':  ('v2_strategy_btc_OKX_BTCUSDT.P_2026-05-22_c2fde.xlsx',   'BTC/USDT'),
    'ETH':  ('v2_strategy_eth_OKX_ETHUSDT.P_2026-05-22_0f7c1.xlsx',   'ETH/USDT'),
    'SOL':  ('v2_strategy_sol_OKX_SOLUSDT.P_2026-05-22_6d6ed.xlsx',   'SOL/USDT'),
    'DOGE': ('v2_strategy_doge_OKX_DOGEUSDT.P_2026-05-22_8856b.xlsx', 'DOGE/USDT'),
}

CAPITAL = 20000
TF      = '2h'

NS     = [7, 10]
THS    = [0.25, 0.30, 0.35, 0.40]

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
    df['ts'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[us]')
    df = df.set_index('ts').sort_index()
    ohlcv_cache[key] = df
    print('done')
    return df

def compute_er(df, n):
    c = df['close']
    return c.diff(n).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)

def get_er_at_entry(trades, ohlcv, n, lag):
    """
    lag=1: 取入场时刻前一根已收盘K线的ER（对应Pine Script里的[1]）
    lag=0: 取入场时刻当前K线的ER（对应Pine Script里不加[1]）
    """
    er = compute_er(ohlcv, n)
    idx = ohlcv.index
    vals = []
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        target = pos - lag
        vals.append(er.iloc[target] if target >= 0 else np.nan)
    return np.array(vals)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_version(version_name, files):
    print(f'\n拉取 {version_name} 数据...')
    sym_data = {}
    for sym, (fname, ccxt_sym) in files.items():
        trades = load_trades(fname)
        ohlcv  = fetch_ohlcv(ccxt_sym, TF)
        pnl    = trades['pnl'].values * (CAPITAL / 50000)
        # 预计算所有 N × lag 组合
        er_vals = {}
        for n in NS:
            for lag in [0, 1]:
                er_vals[(n, lag)] = get_er_at_entry(trades, ohlcv, n, lag)
        sym_data[sym] = {'pnl': pnl, 'er': er_vals}

    # 合并所有标的
    all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
    all_er  = {k: np.concatenate([d['er'][k] for d in sym_data.values()])
               for k in [(n, lag) for n in NS for lag in [0, 1]]}
    base = all_pnl.sum()
    n_trades = len(all_pnl)

    print(f'\n{"="*70}')
    print(f'{version_name}  {n_trades}笔  基准 {base:+.0f}')
    print(f'{"="*70}')

    # 表头
    th_cols = ''.join(f'  ≥{th:.2f}' for th in THS)
    print(f'\n  {"参数":<18}{th_cols}')
    print('  ' + '-' * (18 + 8 * len(THS)))

    results = {}
    for n in NS:
        for lag in [0, 1]:
            label = f'N={n} {"[1]" if lag else "当前"}'
            er = all_er[(n, lag)]
            row = f'  {label:<18}'
            for th in THS:
                mask = ~np.isnan(er) & (er >= th)
                filtered_pnl = all_pnl[mask].sum()
                lift_pct = (filtered_pnl - base) / abs(base) * 100
                row += f'  {lift_pct:>+5.1f}%'
                results[(n, lag, th)] = {'pnl': filtered_pnl, 'lift': lift_pct,
                                          'n_kept': mask.sum(), 'keep_rate': mask.mean()}
            print(row)
        print('  ' + '-' * (18 + 8 * len(THS)))

    # 过滤率对比
    print(f'\n  {"参数":<18}' + ''.join(f'  ≥{th:.2f}' for th in THS) + '  （保留率）')
    print('  ' + '-' * (18 + 8 * len(THS) + 10))
    for n in NS:
        for lag in [0, 1]:
            label = f'N={n} {"[1]" if lag else "当前"}'
            er = all_er[(n, lag)]
            row = f'  {label:<18}'
            for th in THS:
                mask = ~np.isnan(er) & (er >= th)
                row += f'  {mask.mean()*100:>4.0f}%  '
            print(row)

    # [1] vs 当前 差值
    print(f'\n  [1] vs 当前 差值（正数=当前更好）')
    print(f'  {"N":<18}' + ''.join(f'  ≥{th:.2f}' for th in THS))
    print('  ' + '-' * (18 + 8 * len(THS)))
    for n in NS:
        row = f'  N={n:<16}'
        for th in THS:
            diff = results[(n, 0, th)]['lift'] - results[(n, 1, th)]['lift']
            row += f'  {diff:>+5.1f}%'
        print(row)

    return results, base, all_pnl, all_er

if __name__ == '__main__':
    r_v1, base_v1, pnl_v1, er_v1 = run_version('v1', V1_FILES)
    r_v2, base_v2, pnl_v2, er_v2 = run_version('v2', V2_FILES)

    # v1+v2 合计
    all_pnl = np.concatenate([pnl_v1, pnl_v2])
    all_er  = {k: np.concatenate([er_v1[k], er_v2[k]])
               for k in [(n, lag) for n in NS for lag in [0, 1]]}
    base = all_pnl.sum()

    print(f'\n{"="*70}')
    print(f'v1+v2 合计  {len(all_pnl)}笔  基准 {base:+.0f}')
    print(f'{"="*70}')
    th_cols = ''.join(f'  ≥{th:.2f}' for th in THS)
    print(f'\n  {"参数":<18}{th_cols}')
    print('  ' + '-' * (18 + 8 * len(THS)))
    for n in NS:
        for lag in [0, 1]:
            label = f'N={n} {"[1]" if lag else "当前"}'
            er = all_er[(n, lag)]
            row = f'  {label:<18}'
            for th in THS:
                mask = ~np.isnan(er) & (er >= th)
                lift = (all_pnl[mask].sum() - base) / abs(base) * 100
                row += f'  {lift:>+5.1f}%'
            print(row)
        print('  ' + '-' * (18 + 8 * len(THS)))

    print(f'\n  [1] vs 当前 差值（正数=当前更好）')
    print(f'  {"N":<18}' + ''.join(f'  ≥{th:.2f}' for th in THS))
    print('  ' + '-' * (18 + 8 * len(THS)))
    for n in NS:
        row = f'  N={n:<16}'
        for th in THS:
            m0 = ~np.isnan(all_er[(n,0)]) & (all_er[(n,0)] >= th)
            m1 = ~np.isnan(all_er[(n,1)]) & (all_er[(n,1)] >= th)
            diff = (all_pnl[m0].sum() - all_pnl[m1].sum()) / abs(base) * 100
            row += f'  {diff:>+5.1f}%'
        print(row)
