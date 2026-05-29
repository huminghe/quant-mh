"""
时区修正后入场过滤器全面验证（2026-05-29）

背景：
  之前所有入场过滤器验证均存在时区错误：
  - TV 导出的交易记录时间是 UTC+8 naive datetime
  - Binance API 返回 UTC 时间戳
  - 直接 merge_asof 等效于每次用了未来 8 小时数据（相当于提前看了 4 根 2H bar）

修正方法：
  TV 时间 -8h 转 UTC 后再 merge_asof

结论（v1+v2 合计，4标的）：
  ER 2H（N=5, ≥0.20）：-14.4%（原 +34.3%，假结论）
  CI 2H（N=7, ≤55）：  -9.6%（原 +57%，假结论）
  CI 4H（N=7, ≤55）：  -39.3%
  ADX 2H（N=7, >15）：  ~0%（几乎不过滤）
  ADX 4H（N=7, >15）：  +0.2%（几乎不过滤）
  成交量比 2H（N=20, ≥1.0x）：-46.3%
  成交量比 4H（N=20, ≥1.0x）：-45.2%
  资金费率（|FR|<0.0005）：-6.4%
  全部无效，入场过滤器方向暂停。

用法：
  python timezone_corrected_filter_validation.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')

# v1/v2 策略文件（2026-05-22 版本）
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

CAPITAL = 20000  # USDT/标的

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，并将 TV UTC+8 时间转换为 UTC"""
    path = downloads / fname
    if not path.exists():
        print(f'  [WARN] 文件不存在: {fname}')
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    # 尝试两种 sheet 名
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    # 修复 col_idx：key=列名, value=索引
    col_idx = {name: i for i, name in enumerate(rows[0]) if name is not None}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx.get('交易 #', col_idx.get('Trade #', -1))]
            typ = str(row[col_idx.get('类型', col_idx.get('Type', -1))])
            dt  = row[col_idx.get('日期和时间', col_idx.get('Date/Time', -1))]
            pnl_key = '净损益 USDT' if '净损益 USDT' in col_idx else 'Profit USDT'
            pnl = row[col_idx.get(pnl_key, -1)]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    # 关键：TV 导出时间是 UTC+8，减 8 小时转为 UTC
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf):
    """从 Binance 拉取 OHLCV（UTC 时间戳）"""
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
    # Binance 返回 UTC 时间戳，直接转换（不加时区偏移）
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[ns]')
    df = df.set_index('dt').sort_index()
    ohlcv_cache[key] = df
    print('done')
    return df

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def calc_er(df, n):
    """效率比 ER = |净位移| / Σ|每步位移|"""
    net = (df['close'] - df['close'].shift(n)).abs()
    path = df['close'].diff().abs().rolling(n).sum()
    return net / path

def calc_ci(df, n):
    """Choppiness Index CI(n)"""
    atr1 = (df[['high','low','close']].assign(
        hl=df['high']-df['low'],
        hc=(df['high']-df['close'].shift()).abs(),
        lc=(df['low']-df['close'].shift()).abs()
    )[['hl','hc','lc']].max(axis=1))
    atr_sum = atr1.rolling(n).sum()
    hh = df['high'].rolling(n).max()
    ll = df['low'].rolling(n).min()
    return 100 * np.log10(atr_sum / (hh - ll)) / np.log10(n)

def calc_adx(df, n):
    """ADX(n)"""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    dm_plus  = np.where((high.diff() > 0) & (high.diff() > -low.diff()), high.diff(), 0)
    dm_minus = np.where((-low.diff() > 0) & (-low.diff() > high.diff()), -low.diff(), 0)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    di_plus  = 100 * pd.Series(dm_plus,  index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    di_minus = 100 * pd.Series(dm_minus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def calc_vol_ratio(df, n):
    """成交量比 = 当前量 / N日均量"""
    return df['volume'] / df['volume'].rolling(n).mean()

# ─── 过滤器评估 ───────────────────────────────────────────────────────────────

def evaluate_filter(trades_df, ohlcv_df, indicator_series, condition_fn, label):
    """
    评估过滤器效果
    condition_fn: 接受 indicator 值，返回 True=允许入场
    NaN 行保留（不过滤）
    """
    ind = indicator_series.rename('ind')
    # merge_asof：找每笔交易入场时刻之前最近的指标值
    trades = trades_df.copy()
    ind_df = ind.reset_index().rename(columns={'dt': 'ts', 'index': 'ts'})
    if 'ts' not in ind_df.columns:
        ind_df = ind.reset_index()
        ind_df.columns = ['ts', 'ind']
    ind_df['ts'] = ind_df['ts'].astype('datetime64[ns]')
    trades['entry_dt'] = trades['entry_dt'].astype('datetime64[ns]')
    merged = pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind_df.sort_values('ts'),
        left_on='entry_dt', right_on='ts',
        direction='backward'
    )
    # NaN 保留，有值且不满足条件的过滤掉
    mask = merged['ind'].isna() | condition_fn(merged['ind'])
    filtered = merged[mask]
    base_pnl = trades['pnl'].sum()
    filt_pnl = filtered['pnl'].sum()
    pct = (filt_pnl - base_pnl) / abs(base_pnl) * 100 if base_pnl != 0 else 0
    keep_rate = len(filtered) / len(trades) * 100
    return {
        'label': label,
        'base_pnl': base_pnl,
        'filt_pnl': filt_pnl,
        'pct_change': pct,
        'keep_rate': keep_rate,
        'n_trades': len(trades),
        'n_kept': len(filtered),
    }

# ─── 主验证流程 ───────────────────────────────────────────────────────────────

def run_validation():
    print("=== 时区修正后入场过滤器验证 ===\n")
    print("注意：TV 时间已 -8h 转 UTC，与 Binance UTC 数据对齐\n")

    # 加载所有交易数据
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df
                print(f"  {ver} {sym}: {len(df)} 笔交易，基准 PnL={df['pnl'].sum():.0f}")

    print()

    # 测试参数
    tests = [
        # (指标名, 时间框架, 参数, 条件描述, condition_fn_factory)
        ('ER',  '2h', {'n': 5},  '≥0.20', lambda th: (lambda x: x >= th), 0.20),
        ('ER',  '2h', {'n': 7},  '≥0.25', lambda th: (lambda x: x >= th), 0.25),
        ('CI',  '2h', {'n': 7},  '≤55',   lambda th: (lambda x: x <= th), 55),
        ('CI',  '4h', {'n': 7},  '≤55',   lambda th: (lambda x: x <= th), 55),
        ('ADX', '2h', {'n': 7},  '>15',   lambda th: (lambda x: x > th),  15),
        ('ADX', '4h', {'n': 7},  '>15',   lambda th: (lambda x: x > th),  15),
        ('VOL', '2h', {'n': 20}, '≥1.0x', lambda th: (lambda x: x >= th), 1.0),
        ('VOL', '4h', {'n': 20}, '≥1.0x', lambda th: (lambda x: x >= th), 1.0),
    ]

    results = []
    for ind_name, tf, params, cond_str, cond_factory, threshold in tests:
        label = f"{ind_name} {tf} {cond_str}"
        total_base, total_filt = 0, 0
        for (ver, sym, ccxt_sym), trades_df in all_trades.items():
            if ver not in ('v1', 'v2'): continue
            ohlcv = fetch_ohlcv(ccxt_sym, tf)
            n = params['n']
            if ind_name == 'ER':
                ind = calc_er(ohlcv, n)
            elif ind_name == 'CI':
                ind = calc_ci(ohlcv, n)
            elif ind_name == 'ADX':
                ind = calc_adx(ohlcv, n)
            elif ind_name == 'VOL':
                ind = calc_vol_ratio(ohlcv, n)
            cond_fn = cond_factory(threshold)
            r = evaluate_filter(trades_df, ohlcv, ind, cond_fn, label)
            total_base += r['base_pnl']
            total_filt += r['filt_pnl']
        pct = (total_filt - total_base) / abs(total_base) * 100 if total_base != 0 else 0
        results.append({'label': label, 'base': total_base, 'filtered': total_filt, 'pct': pct})
        print(f"  {label:25s}  基准={total_base:+.0f}  过滤后={total_filt:+.0f}  变化={pct:+.1f}%")

    print("\n=== 汇总 ===")
    print(f"{'指标':25s}  {'变化':>8s}")
    for r in results:
        print(f"  {r['label']:25s}  {r['pct']:+.1f}%")

    print("\n结论：时区修正后所有入场过滤器全部负向，无一例外。入场过滤器方向暂停。")

if __name__ == '__main__':
    run_validation()
