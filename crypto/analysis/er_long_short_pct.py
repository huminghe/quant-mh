"""
ER 过滤器：多空拆分 + 百分位 ER vs 固定阈值（2026-05-29）

验证：
1. ER 过滤效果多空拆分（多头 vs 空头是否对称）
2. 百分位 ER（滚动 percentrank）vs 固定阈值

用法：
  python er_long_short_pct.py
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

CAPITAL = 20000
TF      = '2h'
N       = 7       # v1/v2 最优
TH      = 0.35    # 固定阈值
PCT_THS = [40, 50, 55, 60]   # 百分位阈值（percentrank 0-100）
PCT_WIN = 100     # 百分位计算窗口

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
                # 从"多头进场"/"空头进场"判断方向
                by_num[num]['direction'] = 'long' if '多头' in typ else 'short'
            elif '出场' in typ and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d and 'direction' in d]
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

def compute_er_pct(df, n, window):
    """滚动百分位排名（0-100），window 根 K 线内的相对位置"""
    er = compute_er(df, n)
    # pandas percentrank 等价：rolling rank / window * 100
    pct = er.rolling(window).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else np.nan,
        raw=True
    )
    return er, pct

def get_vals_at_entry(trades, ohlcv, *series_list):
    """取每笔交易入场时刻对应的指标值（前一根已收盘K线）"""
    idx = ohlcv.index
    results = [[] for _ in series_list]
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        for i, s in enumerate(series_list):
            results[i].append(s.iloc[pos] if pos >= 0 else np.nan)
    return [np.array(r) for r in results]

# ─── 打印工具 ─────────────────────────────────────────────────────────────────

def lift_str(filtered_pnl, base):
    lift = filtered_pnl - base
    pct  = lift / abs(base) * 100 if base != 0 else 0
    return f'{pct:>+6.1f}%'

def print_long_short_table(label, pnl, er, direction, base_long, base_short, base_all):
    """多空拆分表"""
    long_mask  = direction == 'long'
    short_mask = direction == 'short'

    print(f'\n  [{label}]')
    header = f'  {"过滤条件":<22}  {"全部":>8}  {"多头":>8}  {"空头":>8}  {"多头保留率":>8}  {"空头保留率":>8}'
    print(header)
    print('  ' + '-' * 72)

    # 无过滤基准
    print(f'  {"无过滤":<22}  {base_all:>+8.0f}  {base_long:>+8.0f}  {base_short:>+8.0f}  {"100%":>8}  {"100%":>8}')
    print('  ' + '-' * 72)

    # 固定阈值
    for th in [0.25, 0.30, 0.35, 0.40]:
        mask = ~np.isnan(er) & (er >= th)
        all_f  = pnl[mask].sum()
        long_f = pnl[mask & long_mask].sum()
        short_f= pnl[mask & short_mask].sum()
        lr = mask[long_mask].mean() * 100
        sr = mask[short_mask].mean() * 100
        row = (f'  {f"ER≥{th:.2f}":<22}'
               f'  {lift_str(all_f, base_all):>8}'
               f'  {lift_str(long_f, base_long):>8}'
               f'  {lift_str(short_f, base_short):>8}'
               f'  {lr:>7.0f}%'
               f'  {sr:>7.0f}%')
        print(row)

def print_pct_table(label, pnl, er_pct, direction, base_long, base_short, base_all):
    """百分位 ER 表"""
    long_mask  = direction == 'long'
    short_mask = direction == 'short'

    print(f'\n  [{label} — 百分位 ER（窗口={PCT_WIN}）]')
    header = f'  {"过滤条件":<22}  {"全部":>8}  {"多头":>8}  {"空头":>8}  {"多头保留率":>8}  {"空头保留率":>8}'
    print(header)
    print('  ' + '-' * 72)

    for th in PCT_THS:
        mask = ~np.isnan(er_pct) & (er_pct >= th)
        all_f  = pnl[mask].sum()
        long_f = pnl[mask & long_mask].sum()
        short_f= pnl[mask & short_mask].sum()
        lr = mask[long_mask].mean() * 100
        sr = mask[short_mask].mean() * 100
        row = (f'  {f"ER_pct≥{th}":<22}'
               f'  {lift_str(all_f, base_all):>8}'
               f'  {lift_str(long_f, base_long):>8}'
               f'  {lift_str(short_f, base_short):>8}'
               f'  {lr:>7.0f}%'
               f'  {sr:>7.0f}%')
        print(row)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_versions = {}

    for ver, sym_files in VERSION_FILES.items():
        print(f'\n加载 {ver}...')
        sym_data = {}
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            ohlcv  = fetch_ohlcv(ccxt_sym, TF)
            pnl    = trades['pnl'].values * (CAPITAL / 50000)
            er_series, er_pct_series = compute_er_pct(ohlcv, N, PCT_WIN)
            er_vals, er_pct_vals = get_vals_at_entry(trades, ohlcv, er_series, er_pct_series)
            sym_data[sym] = {
                'pnl': pnl,
                'er': er_vals,
                'er_pct': er_pct_vals,
                'direction': trades['direction'].values,
            }
        all_versions[ver] = sym_data

    # ── 各版本输出 ──
    for ver, sym_data in all_versions.items():
        all_pnl  = np.concatenate([d['pnl']       for d in sym_data.values()])
        all_er   = np.concatenate([d['er']         for d in sym_data.values()])
        all_pct  = np.concatenate([d['er_pct']     for d in sym_data.values()])
        all_dir  = np.concatenate([d['direction']  for d in sym_data.values()])

        long_mask  = all_dir == 'long'
        short_mask = all_dir == 'short'
        base_all   = all_pnl.sum()
        base_long  = all_pnl[long_mask].sum()
        base_short = all_pnl[short_mask].sum()

        print(f'\n{"="*72}')
        print(f'{ver}  {len(all_pnl)}笔  基准{base_all:+.0f}'
              f'  多头{len(all_pnl[long_mask])}笔({base_long:+.0f})'
              f'  空头{len(all_pnl[short_mask])}笔({base_short:+.0f})')
        print(f'{"="*72}')

        print_long_short_table(f'{ver} 固定阈值 N={N}', all_pnl, all_er, all_dir,
                               base_long, base_short, base_all)
        print_pct_table(f'{ver} N={N}', all_pnl, all_pct, all_dir,
                        base_long, base_short, base_all)

    # ── v1+v2 合计 ──
    all_pnl = np.concatenate([d['pnl']      for v in all_versions.values() for d in v.values()])
    all_er  = np.concatenate([d['er']       for v in all_versions.values() for d in v.values()])
    all_pct = np.concatenate([d['er_pct']   for v in all_versions.values() for d in v.values()])
    all_dir = np.concatenate([d['direction']for v in all_versions.values() for d in v.values()])

    long_mask  = all_dir == 'long'
    short_mask = all_dir == 'short'
    base_all   = all_pnl.sum()
    base_long  = all_pnl[long_mask].sum()
    base_short = all_pnl[short_mask].sum()

    print(f'\n{"="*72}')
    print(f'v1+v2 合计  {len(all_pnl)}笔  基准{base_all:+.0f}'
          f'  多头{long_mask.sum()}笔({base_long:+.0f})'
          f'  空头{short_mask.sum()}笔({base_short:+.0f})')
    print(f'{"="*72}')

    print_long_short_table('v1+v2 固定阈值 N=7', all_pnl, all_er, all_dir,
                           base_long, base_short, base_all)
    print_pct_table('v1+v2 N=7', all_pnl, all_pct, all_dir,
                    base_long, base_short, base_all)

    # ── 固定阈值 vs 百分位最优对比汇总 ──
    print(f'\n{"="*72}')
    print('固定阈值 vs 百分位 ER 最优对比（v1+v2 合计）')
    print(f'{"="*72}')
    best_fixed = max(
        (all_pnl[~np.isnan(all_er) & (all_er >= th)].sum(), th)
        for th in [0.25, 0.30, 0.35, 0.40]
    )
    best_pct = max(
        (all_pnl[~np.isnan(all_pct) & (all_pct >= th)].sum(), th)
        for th in PCT_THS
    )
    print(f'  固定阈值最优: ER≥{best_fixed[1]:.2f}  {best_fixed[0]:+.0f}  ({lift_str(best_fixed[0], base_all).strip()})')
    print(f'  百分位最优:   ER_pct≥{best_pct[1]}   {best_pct[0]:+.0f}  ({lift_str(best_pct[0], base_all).strip()})')
    print(f'  百分位 vs 固定阈值差值: {(best_pct[0]-best_fixed[0])/abs(base_all)*100:+.1f}%')
