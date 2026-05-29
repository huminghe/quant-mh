"""
Hurst 指数入场过滤验证（2026-05-27）

用 DFA（去趋势波动分析）计算 Hurst 指数：
- H > 0.5：正自相关，趋势持续性（适合趋势跟踪）
- H = 0.5：随机游走
- H < 0.5：负自相关，均值回归

DFA 方法：
1. 对收益率序列做累积和（profile）
2. 分段线性拟合，计算各段残差的 RMS
3. log(RMS) vs log(段长) 的斜率即为 H

验证：
1. Hurst 单独作为入场过滤（多个窗口 × 多个阈值）
2. 与 ER≥0.3 对比
3. ER + Hurst 组合

用法：
  python hurst_filter_validation.py
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

CAPITAL    = 20000
VERSION_TF = {'v1': '2h', 'v2': '2h', 'v3_205m': '1h', 'v3_3h': '1h'}

# Hurst 参数
HURST_WINDOWS = [63, 100]        # 计算窗口（根 K 线数）
HURST_THS     = [0.50, 0.52, 0.55, 0.58]  # 阈值（H > th 才入场）

# ER 基准
ER_N  = 10
ER_TH = 0.3

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
    return df

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def _dfa_hurst(x):
    """对一段序列计算 DFA Hurst 指数（向量化内层循环）"""
    n = len(x)
    if n < 20: return np.nan
    profile = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(4), np.log10(n // 4), 8).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 3: return np.nan
    flucts = []
    for s in scales:
        n_segs = n // s
        if n_segs < 2: continue
        # 把所有段堆成矩阵，一次性做最小二乘去趋势
        segs = profile[:n_segs * s].reshape(n_segs, s)   # (n_segs, s)
        t = np.arange(s, dtype=float)
        # 解析解：线性拟合残差 RMS
        t_mean = t.mean()
        t_var  = ((t - t_mean) ** 2).sum()
        seg_mean = segs.mean(axis=1, keepdims=True)       # (n_segs, 1)
        slope = ((segs - seg_mean) * (t - t_mean)).sum(axis=1) / t_var  # (n_segs,)
        intercept = seg_mean[:, 0] - slope * t_mean
        trend = slope[:, None] * t[None, :] + intercept[:, None]        # (n_segs, s)
        rms = np.sqrt(((segs - trend) ** 2).mean(axis=1)).mean()
        flucts.append(rms)
    if len(flucts) < 3: return np.nan
    log_s = np.log(scales[:len(flucts)])
    log_f = np.log(flucts)
    h, _ = np.polyfit(log_s, log_f, 1)
    return h

def compute_hurst(df, window):
    """滚动 DFA Hurst，用收益率序列"""
    ret = df['close'].pct_change().fillna(0).values
    result = np.full(len(ret), np.nan)
    for i in range(window, len(ret)):
        result[i] = _dfa_hurst(ret[i-window:i])
    return pd.Series(result, index=df.index)

def compute_er(df, n=ER_N):
    c = df['close']
    return c.diff(n).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)

def get_val_at_entry(trades, ohlcv, series):
    idx = ohlcv.index
    vals = []
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        vals.append(series.iloc[pos - 1] if pos >= 1 else np.nan)
    return np.array(vals)

# ─── 核心验证 ─────────────────────────────────────────────────────────────────

def run_version(version_name, symbol_files):
    tf = VERSION_TF[version_name]

    sym_data = {}
    for sym, (fname, ccxt_sym) in symbol_files.items():
        trades = load_trades(fname)
        if trades.empty: continue

        pnl   = trades['pnl'].values * (CAPITAL / 50000)
        ohlcv = fetch_ohlcv(ccxt_sym, tf)
        er_vals = get_val_at_entry(trades, ohlcv, compute_er(ohlcv))

        print(f'    计算 {sym} Hurst...', end=' ', flush=True)
        hurst_by_w = {}
        for w in HURST_WINDOWS:
            h_series = compute_hurst(ohlcv, w)
            hurst_by_w[w] = get_val_at_entry(trades, ohlcv, h_series)
        print('done')

        sym_data[sym] = {'pnl': pnl, 'er': er_vals, 'hurst': hurst_by_w}

    if not sym_data: return None

    print(f'\n{"="*72}')
    print(f'策略版本: {version_name}  (时间框架: {tf})')
    print(f'{"="*72}')

    for sym, d in sym_data.items():
        pnl  = d['pnl']
        base = pnl.sum()
        n    = len(pnl)
        er_mask = ~np.isnan(d['er']) & (d['er'] >= ER_TH)
        er_pnl  = pnl[er_mask].sum()

        # Hurst 分布
        for w in HURST_WINDOWS:
            h = d['hurst'][w]
            valid = h[~np.isnan(h)]
            print(f'\n  [{sym}] w={w}  {n}笔  基准{base:+.0f}  ER≥{ER_TH} {er_pnl-base:+.0f}({(er_pnl-base)/abs(base)*100:+.1f}%)'
                  f'  H分布: 中位={np.median(valid):.3f} >0.5占{(valid>0.5).mean()*100:.0f}%')
            th_header = ''.join(f'  H>{th:.2f}' for th in HURST_THS)
            print(f'  {"":>4}{th_header}   ER+H最优')
            print('  ' + '-' * (4 + 9 * len(HURST_THS) + 10))
            row = '  单独'
            best_combo = -np.inf
            for th in HURST_THS:
                mask = ~np.isnan(h) & (h > th)
                lift = pnl[mask].sum() - base
                pct  = lift / abs(base) * 100
                row += f'  {pct:>+5.1f}%'
                combo = pnl[er_mask & mask].sum() - base
                if combo > best_combo:
                    best_combo = combo
                    best_combo_pct = combo / abs(base) * 100
            row += f'   {best_combo_pct:>+5.1f}%'
            print(row)

    # 版本合计
    all_pnl   = np.concatenate([d['pnl'] for d in sym_data.values()])
    all_er    = np.concatenate([d['er']  for d in sym_data.values()])
    all_hurst = {w: np.concatenate([d['hurst'][w] for d in sym_data.values()]) for w in HURST_WINDOWS}
    base_total = all_pnl.sum()
    er_mask_all = ~np.isnan(all_er) & (all_er >= ER_TH)
    er_total    = all_pnl[er_mask_all].sum()

    print(f'\n  [{version_name} 合计]  {len(all_pnl)}笔  基准{base_total:+.0f}  ER≥{ER_TH} {er_total-base_total:+.0f}({(er_total-base_total)/abs(base_total)*100:+.1f}%)')
    for w in HURST_WINDOWS:
        h = all_hurst[w]
        valid = h[~np.isnan(h)]
        print(f'  w={w}  H中位={np.median(valid):.3f}  >0.5占{(valid>0.5).mean()*100:.0f}%')
        th_header = ''.join(f'  H>{th:.2f}' for th in HURST_THS)
        print(f'  {"":>4}{th_header}   ER+H最优')
        print('  ' + '-' * (4 + 9 * len(HURST_THS) + 10))
        row = '  单独'
        best_combo = -np.inf
        for th in HURST_THS:
            mask = ~np.isnan(h) & (h > th)
            lift = all_pnl[mask].sum() - base_total
            pct  = lift / abs(base_total) * 100
            row += f'  {pct:>+5.1f}%'
            combo = all_pnl[er_mask_all & mask].sum() - base_total
            if combo > best_combo:
                best_combo = combo
                best_combo_pct = combo / abs(base_total) * 100
        row += f'   {best_combo_pct:>+5.1f}%'
        print(row)

    return {
        'version': version_name,
        'base': base_total, 'er_pnl': er_total,
        'all_pnl': all_pnl, 'all_er': all_er, 'all_hurst': all_hurst,
    }

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []
    for ver, sym_files in VERSION_FILES.items():
        r = run_version(ver, sym_files)
        if r: all_results.append(r)

    total_base = sum(r['base'] for r in all_results)
    total_er   = sum(r['er_pnl'] for r in all_results)

    print(f'\n{"="*72}')
    print('全版本合计')
    print(f'{"="*72}')
    print(f'  基准: {total_base:+.0f}  ER≥{ER_TH}: {total_er:+.0f}  提升 {total_er-total_base:+.0f} ({(total_er-total_base)/abs(total_base)*100:+.1f}%)')

    all_pnl   = np.concatenate([r['all_pnl'] for r in all_results])
    all_er    = np.concatenate([r['all_er']  for r in all_results])
    all_hurst = {w: np.concatenate([r['all_hurst'][w] for r in all_results]) for w in HURST_WINDOWS}
    er_mask   = ~np.isnan(all_er) & (all_er >= ER_TH)

    for w in HURST_WINDOWS:
        h = all_hurst[w]
        valid = h[~np.isnan(h)]
        print(f'\n  w={w}  H中位={np.median(valid):.3f}  >0.5占{(valid>0.5).mean()*100:.0f}%  NaN占{np.isnan(h).mean()*100:.0f}%')
        th_header = ''.join(f'  H>{th:.2f}' for th in HURST_THS)
        print(f'  {"":>4}{th_header}   ER+H最优')
        print('  ' + '-' * (4 + 9 * len(HURST_THS) + 10))
        row = '  单独'
        best_combo = -np.inf
        for th in HURST_THS:
            mask = ~np.isnan(h) & (h > th)
            lift = all_pnl[mask].sum() - total_base
            pct  = lift / abs(total_base) * 100
            row += f'  {pct:>+5.1f}%'
            combo = all_pnl[er_mask & mask].sum() - total_base
            if combo > best_combo:
                best_combo = combo
                best_combo_pct = combo / abs(total_base) * 100
        row += f'   {best_combo_pct:>+5.1f}%'
        print(row)
