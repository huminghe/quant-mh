"""
ACF（自相关系数）入场过滤验证（2026-05-27）

ACF(lag) = 价格收益率序列与其滞后 lag 期的 Pearson 相关系数
- ACF > 0：正自相关，价格有延续性（趋势）
- ACF ≈ 0：随机游走
- ACF < 0：负自相关，均值回归

验证：
1. ACF 单独作为入场过滤（多个 lag × 多个阈值）
2. 与 ER≥0.3 (2H, N=10) 对比
3. ER + ACF 组合是否优于单独使用

用法：
  python acf_filter_validation.py
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

CAPITAL   = 20000

# 各版本要扫描的时间框架
VERSION_TFS = {
    'v1':  ['4h', '8h', '1d'],
    'v2':  ['4h', '8h', '1d'],
}

# ACF 参数
ACF_WINDOW = 20
ACF_LAGS   = [1, 3, 5]
ACF_THS    = [0.05, 0.10, 0.15, 0.20]

# ER 基准（v1/v2 用 2H N=10）
ER_N  = 10
ER_TH = 0.3
ER_TF = '2h'

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

def compute_acf(df, lag, window=ACF_WINDOW):
    """滚动 ACF：在 window 根 K 线的收益率序列上计算 lag 阶自相关"""
    ret = df['close'].pct_change()
    def _acf(x):
        if len(x) < lag + 2: return np.nan
        x = x - x.mean()
        c0 = np.dot(x, x)
        if c0 == 0: return np.nan
        return np.dot(x[:-lag], x[lag:]) / c0
    return ret.rolling(window + lag).apply(_acf, raw=True)

def compute_er(df, n=ER_N):
    c = df['close']
    return c.diff(n).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)

def get_val_at_entry(trades, ohlcv, series):
    """取进场时刻对应 K 线的 [1]（上一根已收盘）指标值"""
    idx = ohlcv.index
    vals = []
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        vals.append(series.iloc[pos - 1] if pos >= 1 else np.nan)
    return np.array(vals)

# ─── 核心验证 ─────────────────────────────────────────────────────────────────

def run_version_tf(version_name, symbol_files, acf_tf):
    """对一个版本 + 一个 ACF 时间框架，输出合计结果"""
    sym_data = {}
    for sym, (fname, ccxt_sym) in symbol_files.items():
        trades = load_trades(fname)
        if trades.empty: continue

        pnl = trades['pnl'].values * (CAPITAL / 50000)

        # ER 固定用 2H
        ohlcv_er = fetch_ohlcv(ccxt_sym, ER_TF)
        er_vals  = get_val_at_entry(trades, ohlcv_er, compute_er(ohlcv_er))

        # ACF 用指定 tf
        ohlcv_acf  = fetch_ohlcv(ccxt_sym, acf_tf)
        acf_by_lag = {
            lag: get_val_at_entry(trades, ohlcv_acf, compute_acf(ohlcv_acf, lag))
            for lag in ACF_LAGS
        }
        sym_data[sym] = {'pnl': pnl, 'er': er_vals, 'acf': acf_by_lag}

    if not sym_data: return None

    all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
    all_er  = np.concatenate([d['er']  for d in sym_data.values()])
    all_acf = {
        lag: np.concatenate([d['acf'][lag] for d in sym_data.values()])
        for lag in ACF_LAGS
    }
    base      = all_pnl.sum()
    er_mask   = ~np.isnan(all_er) & (all_er >= ER_TH)
    er_pnl    = all_pnl[er_mask].sum()

    print(f'\n  [{version_name} ACF={acf_tf}]  {len(all_pnl)}笔  基准{base:+.0f}  ER(2H)≥{ER_TH} {er_pnl-base:+.0f}({(er_pnl-base)/abs(base)*100:+.1f}%)')
    th_header = ''.join(f'  ≥{th:.2f}' for th in ACF_THS)
    print(f'  {"lag":>5}{th_header}   ER+ACF最优')
    print('  ' + '-' * (5 + 8 * len(ACF_THS) + 12))
    for lag in ACF_LAGS:
        acf = all_acf[lag]
        row = f'  lag={lag}'
        best_combo = -np.inf
        for th in ACF_THS:
            mask = ~np.isnan(acf) & (acf >= th)
            lift = all_pnl[mask].sum() - base
            pct  = lift / abs(base) * 100
            row += f'  {pct:>+5.1f}%'
            combo = all_pnl[er_mask & mask].sum() - base
            if combo > best_combo:
                best_combo = combo
                best_combo_pct = combo / abs(base) * 100
        row += f'   {best_combo_pct:>+5.1f}%'
        print(row)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    versions_to_run = {k: v for k, v in VERSION_FILES.items() if k in VERSION_TFS}

    for ver, sym_files in versions_to_run.items():
        print(f'\n{"="*72}')
        print(f'策略版本: {ver}')
        print(f'{"="*72}')
        for tf in VERSION_TFS[ver]:
            run_version_tf(ver, sym_files, tf)

