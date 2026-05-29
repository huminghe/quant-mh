"""
ER(10) 2H ≥0.3 入场过滤 — 全策略版本验证（2026-05-27）

验证 ER≥0.3 (2H) 在 v1/v2/v3_205m/v3_3h 上的效果，
与 CI≤50 (2H) 对比，输出各版本合计提升。

用法：
  python er_filter_all_versions.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')

# 各版本文件配置
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
    'v3_205m': {
        'BTC':  ('v3_205m_strategy_btc_OKX_BTCUSDT.P_2026-05-22_72b80.xlsx',  'BTC/USDT'),
        'ETH':  ('v3_205m_strategy_eth_OKX_ETHUSDT.P_2026-05-22_95b14.xlsx',  'ETH/USDT'),
        'SOL':  ('v3_205m_strategy_sol_OKX_SOLUSDT.P_2026-05-22_53a15.xlsx',  'SOL/USDT'),
        'DOGE': ('v3_205m_strategy_doge_OKX_DOGEUSDT.P_2026-05-22_80858.xlsx','DOGE/USDT'),
    },
    'v3_3h': {
        'ETH':  ('v3_3h_strategy_eth_OKX_ETHUSDT.P_2026-05-22_9c87c.xlsx',   'ETH/USDT'),
        'SOL':  ('v5_3h_strategy_sol_OKX_SOLUSDT.P_2026-05-22_dba6c.xlsx',   'SOL/USDT'),
    },
}

CAPITAL = 20000
CI_N    = 10
CI_TH   = 50

ER_NS          = [5, 7, 10, 14, 20]
ER_THRESHOLDS  = [0.2, 0.25, 0.3, 0.35, 0.4]

# 各版本使用的时间框架
VERSION_TF = {
    'v1':      '2h',
    'v2':      '2h',
    'v3_205m': '1h',
    'v3_3h':   '1h',
}

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
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
    print(f'  拉取 {symbol} {tf} K线...')
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    all_bars = []
    since = ex.parse8601('2019-01-01T00:00:00Z')
    limit = 1000  # Binance 实际最大返回 1000 根
    while True:
        bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not bars: break
        all_bars.extend(bars)
        if len(bars) < limit: break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[us]')
    df = df.set_index('ts').sort_index()
    ohlcv_cache[key] = df
    return df

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def compute_er(df, n):
    c = df['close']
    direction  = c.diff(n).abs()
    volatility = c.diff().abs().rolling(n).sum()
    return direction / volatility.replace(0, np.nan)

def compute_ci(df, n=CI_N):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum  = tr.rolling(n).sum()
    hh       = df['high'].rolling(n).max()
    ll       = df['low'].rolling(n).min()
    hl_range = hh - ll
    return (100 * np.log10(atr_sum / hl_range.replace(0, np.nan))
            / np.log10(n)).where(hl_range > 0)

def get_indicator_at_entry(trades, ohlcv, indicator_series):
    """在每笔交易进场时刻，取对应时间框架 K 线的指标值（[1]，上一根已收盘）"""
    idx = ohlcv.index
    vals = []
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        if pos >= 1:
            vals.append(indicator_series.iloc[pos - 1])
        else:
            vals.append(np.nan)
    return np.array(vals)

# ─── 核心验证 ─────────────────────────────────────────────────────────────────

def run_version(version_name, symbol_files):
    tf = VERSION_TF[version_name]

    # 收集各标的数据
    sym_data = {}  # sym -> {'pnl', 'er_by_n', 'ci'}
    for sym, (fname, ccxt_sym) in symbol_files.items():
        trades = load_trades(fname)
        if trades.empty:
            continue
        pnl = trades['pnl'].values * (CAPITAL / 50000)

        ohlcv = fetch_ohlcv(ccxt_sym, tf)
        # CI 用 2H（v3 系列也用 2H 做 CI 基准对比）
        ohlcv_2h = fetch_ohlcv(ccxt_sym, '2h') if tf != '2h' else ohlcv
        ci_series = compute_ci(ohlcv_2h)
        ci_vals = get_indicator_at_entry(trades, ohlcv_2h, ci_series)

        er_by_n = {}
        for n in ER_NS:
            er_series = compute_er(ohlcv, n)
            er_by_n[n] = get_indicator_at_entry(trades, ohlcv, er_series)

        sym_data[sym] = {'pnl': pnl, 'er_by_n': er_by_n, 'ci': ci_vals}

    if not sym_data:
        return None

    print(f'\n{"="*70}')
    print(f'策略版本: {version_name}  (ER 时间框架: {tf})')
    print(f'{"="*70}')

    # ── 逐标的明细表 ──────────────────────────────────────────────────────────
    # 对每个标的，输出 N × 阈值 的提升矩阵
    for sym, d in sym_data.items():
        pnl  = d['pnl']
        base = pnl.sum()
        n_total = len(pnl)
        ci_mask = ~np.isnan(d['ci']) & (d['ci'] <= CI_TH)
        ci_pnl  = pnl[ci_mask].sum()

        print(f'\n  [{sym}]  {n_total} 笔  基准 {base:+.0f}  CI≤50(2H) {ci_pnl-base:+.0f}({(ci_pnl-base)/abs(base)*100:+.1f}%)')
        # 表头：阈值
        header = f'  {"N":>4}' + ''.join(f'  ≥{th:.2f}' for th in ER_THRESHOLDS)
        print(header)
        print('  ' + '-' * (4 + 8 * len(ER_THRESHOLDS)))
        for n in ER_NS:
            er = d['er_by_n'][n]
            row = f'  N={n:>2}'
            for th in ER_THRESHOLDS:
                mask = ~np.isnan(er) & (er >= th)
                lift = pnl[mask].sum() - base
                pct  = lift / abs(base) * 100
                row += f'  {pct:>+5.1f}%'
            print(row)

    # ── 版本合计 ──────────────────────────────────────────────────────────────
    all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
    all_ci  = np.concatenate([d['ci']  for d in sym_data.values()])
    all_er  = {n: np.concatenate([d['er_by_n'][n] for d in sym_data.values()]) for n in ER_NS}

    base_total = all_pnl.sum()
    ci_total   = all_pnl[~np.isnan(all_ci) & (all_ci <= CI_TH)].sum()
    n_total    = len(all_pnl)

    print(f'\n  [{version_name} 合计]  {n_total} 笔  基准 {base_total:+.0f}  CI≤50(2H) {ci_total-base_total:+.0f}({(ci_total-base_total)/abs(base_total)*100:+.1f}%)')
    header = f'  {"N":>4}' + ''.join(f'  ≥{th:.2f}' for th in ER_THRESHOLDS)
    print(header)
    print('  ' + '-' * (4 + 8 * len(ER_THRESHOLDS)))
    for n in ER_NS:
        er = all_er[n]
        row = f'  N={n:>2}'
        for th in ER_THRESHOLDS:
            mask = ~np.isnan(er) & (er >= th)
            lift = all_pnl[mask].sum() - base_total
            pct  = lift / abs(base_total) * 100
            row += f'  {pct:>+5.1f}%'
        print(row)

    # 返回供全版本汇总用
    return {
        'version': version_name,
        'base': base_total,
        'ci_pnl': ci_total,
        'n_total': n_total,
        'all_pnl': all_pnl,
        'all_er': all_er,
    }

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []
    for ver, sym_files in VERSION_FILES.items():
        result = run_version(ver, sym_files)
        if result:
            all_results.append(result)

    # 全版本合计
    total_base = sum(r['base'] for r in all_results)
    total_ci   = sum(r['ci_pnl'] for r in all_results)

    # 拼接所有版本的 pnl 和 er（注意：不同版本 N 对应不同 tf，不能直接合并比较，
    # 这里只做各版本独立最优的汇总）
    print(f'\n{"="*70}')
    print('全版本合计（各版本独立最优，提升% vs 各版本基准之和）')
    print(f'{"="*70}')
    print(f'  合计基准: {total_base:+.0f}')
    print(f'  CI≤50 (2H): {total_ci:+.0f}  提升 {total_ci-total_base:+.0f} ({(total_ci-total_base)/abs(total_base)*100:+.1f}%)')
    print()
    header = f'  {"N":>4}' + ''.join(f'  ≥{th:.2f}' for th in ER_THRESHOLDS)
    print(header)
    print('  ' + '-' * (4 + 8 * len(ER_THRESHOLDS)))
    for n in ER_NS:
        row = f'  N={n:>2}'
        for th in ER_THRESHOLDS:
            total_filtered = 0
            for r in all_results:
                er = r['all_er'][n]
                pnl = r['all_pnl']
                mask = ~np.isnan(er) & (er >= th)
                total_filtered += pnl[mask].sum()
            lift = total_filtered - total_base
            pct  = lift / abs(total_base) * 100
            row += f'  {pct:>+5.1f}%'
        print(row)
