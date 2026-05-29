"""
ADX 入场过滤 + 成交量确认 验证（2026-05-29）

验证内容：
1. ADX 入场过滤：ADX > 阈值才开仓（多阈值 × 多时间框架 × 全版本）
2. 成交量确认：当前量 > N日均量 × 倍数（多倍数 × 多时间框架 × 全版本）
3. ER + ADX 组合
4. ER + 成交量组合

用法：
  python adx_volume_filter_validation.py
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

# ADX 参数
ADX_TFS  = ['2h', '4h', '8h', '1d']
ADX_THS  = [15, 20, 25, 30]

# 成交量参数（量比 = 当前量 / N日均量）
VOL_TFS    = ['2h', '4h', '8h', '1d']
VOL_RATIOS = [1.2, 1.5, 2.0, 3.0]
VOL_MA_N   = 20  # 均量窗口

# ER 基准参数
ER_N  = 10
ER_TH = 0.25  # 全版本最优统一参数

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

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def compute_adx(df, period=14):
    """计算 ADX（Wilder 平滑）"""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    di_p = 100 * dm_p.ewm(alpha=1/period, adjust=False).mean() / atr
    di_m = 100 * dm_m.ewm(alpha=1/period, adjust=False).mean() / atr
    dx   = (100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10))
    adx  = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_vol_ratio(df, n=VOL_MA_N):
    """成交量比 = 当前量 / N根均量"""
    vol_ma = df['volume'].rolling(n).mean()
    return df['volume'] / vol_ma.replace(0, np.nan)

def compute_er(df, n=ER_N):
    """效率比 ER"""
    c = df['close']
    return c.diff(n).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)

def get_val_at_entry(trades, ohlcv, series):
    """取每笔交易入场时刻对应的指标值（用前一根已收盘K线，避免前瞻）"""
    idx = ohlcv.index
    vals = []
    for entry_dt in trades['entry_dt']:
        pos = idx.searchsorted(entry_dt, side='right') - 1
        vals.append(series.iloc[pos - 1] if pos >= 1 else np.nan)
    return np.array(vals)

# ─── 结果打印 ─────────────────────────────────────────────────────────────────

def fmt_lift(pnl_filtered, base):
    lift = pnl_filtered - base
    pct  = lift / abs(base) * 100 if base != 0 else 0
    return f'{pct:>+6.1f}%'

def print_filter_table(label_row, label_col, row_vals, col_vals,
                       pnl_arr, base, er_mask, title):
    """通用过滤结果表：行=时间框架，列=阈值"""
    col_w = 9
    header = f'  {"":12}' + ''.join(f'{str(v):>{col_w}}' for v in col_vals) + f'  {"ER+最优":>{col_w}}'
    sep    = '  ' + '-' * (12 + col_w * len(col_vals) + col_w + 2)
    print(f'\n  [{title}]  基准 {base:+.0f}  ER≥{ER_TH} {pnl_arr[er_mask].sum()-base:+.0f}({fmt_lift(pnl_arr[er_mask].sum(), base).strip()})')
    print(header)
    print(sep)
    for tf, masks_by_th in row_vals.items():
        row = f'  {tf:<12}'
        best_combo = -np.inf
        for th, mask in masks_by_th.items():
            lift_pct = fmt_lift(pnl_arr[mask].sum(), base)
            row += f'{lift_pct:>{col_w}}'
            combo_pnl = pnl_arr[er_mask & mask].sum()
            if combo_pnl > best_combo:
                best_combo = combo_pnl
                best_combo_pct = fmt_lift(combo_pnl, base)
        row += f'  {best_combo_pct:>{col_w}}'
        print(row)

# ─── 核心验证 ─────────────────────────────────────────────────────────────────

def run_version(version_name, symbol_files):
    tf_base = VERSION_TF[version_name]

    # 预拉所有需要的 K 线
    symbols = list({ccxt_sym for _, ccxt_sym in symbol_files.values()})
    all_tfs = set([tf_base] + ADX_TFS + VOL_TFS)
    for sym in symbols:
        for tf in all_tfs:
            fetch_ohlcv(sym, tf)

    sym_data = {}
    for sym, (fname, ccxt_sym) in symbol_files.items():
        trades = load_trades(fname)
        if trades.empty: continue
        pnl = trades['pnl'].values * (CAPITAL / 50000)

        # ER（用策略自身时间框架）
        ohlcv_base = fetch_ohlcv(ccxt_sym, tf_base)
        er_vals = get_val_at_entry(trades, ohlcv_base, compute_er(ohlcv_base))

        # ADX：各时间框架
        adx_by_tf = {}
        for tf in ADX_TFS:
            ohlcv = fetch_ohlcv(ccxt_sym, tf)
            adx_by_tf[tf] = get_val_at_entry(trades, ohlcv, compute_adx(ohlcv))

        # 成交量比：各时间框架
        vol_by_tf = {}
        for tf in VOL_TFS:
            ohlcv = fetch_ohlcv(ccxt_sym, tf)
            vol_by_tf[tf] = get_val_at_entry(trades, ohlcv, compute_vol_ratio(ohlcv))

        sym_data[sym] = {
            'pnl': pnl, 'er': er_vals,
            'adx': adx_by_tf, 'vol': vol_by_tf,
        }

    if not sym_data: return None

    # 合并所有标的
    all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
    all_er  = np.concatenate([d['er']  for d in sym_data.values()])
    all_adx = {tf: np.concatenate([d['adx'][tf] for d in sym_data.values()]) for tf in ADX_TFS}
    all_vol = {tf: np.concatenate([d['vol'][tf] for d in sym_data.values()]) for tf in VOL_TFS}

    base     = all_pnl.sum()
    er_mask  = ~np.isnan(all_er) & (all_er >= ER_TH)

    print(f'\n{"="*72}')
    print(f'策略版本: {version_name}  ({len(all_pnl)}笔  基准{base:+.0f}  ER≥{ER_TH} {all_pnl[er_mask].sum()-base:+.0f}({fmt_lift(all_pnl[er_mask].sum(), base).strip()}))')
    print(f'{"="*72}')

    # ── ADX 过滤表 ──
    adx_masks = {}
    for tf in ADX_TFS:
        h = all_adx[tf]
        adx_masks[tf] = {th: (~np.isnan(h) & (h > th)) for th in ADX_THS}
    print_filter_table(
        'tf', 'th', adx_masks, ADX_THS,
        all_pnl, base, er_mask,
        f'ADX 入场过滤（ADX>阈值才开仓）  阈值→'
    )
    # 打印列标题补充
    print(f'  {"":12}' + ''.join(f'  ADX>{th}' for th in ADX_THS))

    # ── 成交量过滤表 ──
    vol_masks = {}
    for tf in VOL_TFS:
        v = all_vol[tf]
        vol_masks[tf] = {r: (~np.isnan(v) & (v >= r)) for r in VOL_RATIOS}
    print_filter_table(
        'tf', 'ratio', vol_masks, VOL_RATIOS,
        all_pnl, base, er_mask,
        f'成交量确认（量比≥阈值才开仓）  倍数→'
    )
    print(f'  {"":12}' + ''.join(f'  ≥{r}x' for r in VOL_RATIOS))

    return {
        'version': version_name,
        'base': base, 'er_pnl': all_pnl[er_mask].sum(),
        'all_pnl': all_pnl, 'all_er': all_er,
        'all_adx': all_adx, 'all_vol': all_vol,
    }

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []
    for ver, sym_files in VERSION_FILES.items():
        r = run_version(ver, sym_files)
        if r: all_results.append(r)

    # 全版本合计
    total_pnl = np.concatenate([r['all_pnl'] for r in all_results])
    total_er  = np.concatenate([r['all_er']  for r in all_results])
    total_adx = {tf: np.concatenate([r['all_adx'][tf] for r in all_results]) for tf in ADX_TFS}
    total_vol = {tf: np.concatenate([r['all_vol'][tf] for r in all_results]) for tf in VOL_TFS}

    base    = total_pnl.sum()
    er_mask = ~np.isnan(total_er) & (total_er >= ER_TH)
    er_pnl  = total_pnl[er_mask].sum()

    print(f'\n{"="*72}')
    print(f'全版本合计  {len(total_pnl)}笔  基准{base:+.0f}  ER≥{ER_TH} {er_pnl-base:+.0f}({fmt_lift(er_pnl, base).strip()})')
    print(f'{"="*72}')

    # ADX 全版本合计
    adx_masks = {}
    for tf in ADX_TFS:
        h = total_adx[tf]
        adx_masks[tf] = {th: (~np.isnan(h) & (h > th)) for th in ADX_THS}
    print_filter_table(
        'tf', 'th', adx_masks, ADX_THS,
        total_pnl, base, er_mask,
        'ADX 入场过滤 全版本合计'
    )
    print(f'  {"":12}' + ''.join(f'  ADX>{th}' for th in ADX_THS))

    # 成交量全版本合计
    vol_masks = {}
    for tf in VOL_TFS:
        v = total_vol[tf]
        vol_masks[tf] = {r: (~np.isnan(v) & (v >= r)) for r in VOL_RATIOS}
    print_filter_table(
        'tf', 'ratio', vol_masks, VOL_RATIOS,
        total_pnl, base, er_mask,
        '成交量确认 全版本合计'
    )
    print(f'  {"":12}' + ''.join(f'  ≥{r}x' for r in VOL_RATIOS))

    # 最优组合汇总
    print(f'\n{"="*72}')
    print('最优组合汇总（全版本合计）')
    print(f'{"="*72}')
    print(f'  基准:          {base:+.0f}')
    print(f'  ER≥{ER_TH}:       {er_pnl:+.0f}  ({fmt_lift(er_pnl, base).strip()})')

    # 找 ADX 最优
    best_adx_lift, best_adx_label = -np.inf, ''
    for tf in ADX_TFS:
        h = total_adx[tf]
        for th in ADX_THS:
            mask = ~np.isnan(h) & (h > th)
            pnl  = total_pnl[mask].sum()
            if pnl > best_adx_lift:
                best_adx_lift  = pnl
                best_adx_label = f'ADX>{th} ({tf})'
    print(f'  ADX 单独最优:  {best_adx_lift:+.0f}  ({fmt_lift(best_adx_lift, base).strip()})  [{best_adx_label}]')

    # 找成交量最优
    best_vol_lift, best_vol_label = -np.inf, ''
    for tf in VOL_TFS:
        v = total_vol[tf]
        for r in VOL_RATIOS:
            mask = ~np.isnan(v) & (v >= r)
            pnl  = total_pnl[mask].sum()
            if pnl > best_vol_lift:
                best_vol_lift  = pnl
                best_vol_label = f'量比≥{r}x ({tf})'
    print(f'  成交量单独最优:{best_vol_lift:+.0f}  ({fmt_lift(best_vol_lift, base).strip()})  [{best_vol_label}]')

    # ER + ADX 最优组合
    best_combo_adx, best_combo_adx_label = -np.inf, ''
    for tf in ADX_TFS:
        h = total_adx[tf]
        for th in ADX_THS:
            mask  = er_mask & (~np.isnan(h) & (h > th))
            pnl   = total_pnl[mask].sum()
            if pnl > best_combo_adx:
                best_combo_adx       = pnl
                best_combo_adx_label = f'ER≥{ER_TH} + ADX>{th} ({tf})'
    print(f'  ER+ADX 最优:   {best_combo_adx:+.0f}  ({fmt_lift(best_combo_adx, base).strip()})  [{best_combo_adx_label}]')

    # ER + 成交量最优组合
    best_combo_vol, best_combo_vol_label = -np.inf, ''
    for tf in VOL_TFS:
        v = total_vol[tf]
        for r in VOL_RATIOS:
            mask = er_mask & (~np.isnan(v) & (v >= r))
            pnl  = total_pnl[mask].sum()
            if pnl > best_combo_vol:
                best_combo_vol       = pnl
                best_combo_vol_label = f'ER≥{ER_TH} + 量比≥{r}x ({tf})'
    print(f'  ER+成交量最优: {best_combo_vol:+.0f}  ({fmt_lift(best_combo_vol, base).strip()})  [{best_combo_vol_label}]')
