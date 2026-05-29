"""
资金费率入场过滤验证（2026-05-29）

验证资金费率作为入场过滤器的效果，测试三种逻辑：
1. 中性过滤：资金费率绝对值 < 阈值才入场（避免拥挤交易）
2. 方向过滤：做多时费率 < 阈值（空头拥挤），做空时费率 > -阈值（多头拥挤）
3. 极端反向加分：费率极端反向时（逼空/逼多）允许入场

指标：
- 原始费率（8H 单次）
- 滚动均值（3期/7期，即 24H/56H 均值）
- 滚动均值绝对值（衡量持续偏向程度）

与 ER 的组合效果也一并验证。

用法：
  python funding_rate_filter.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl, time
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
ER_N    = 7
ER_TH   = 0.35

# 资金费率参数
FR_MA_WINDOWS = [3, 7]          # 滚动均值窗口（8H 为单位，3=24H，7=56H）
# 中性过滤阈值（绝对值，单位：小数，0.0001 = 0.01%）
NEUTRAL_THS   = [0.0001, 0.0002, 0.0003, 0.0005]
# 极端阈值（用于"极端反向"逻辑）
EXTREME_TH    = 0.0005

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
    ohlcv_cache[key] = df.set_index('ts').sort_index()
    return ohlcv_cache[key]

fr_cache = {}
def fetch_funding_rate(symbol):
    if symbol in fr_cache: return fr_cache[symbol]
    print(f'  拉取 {symbol} 资金费率...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    all_rates, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        rates = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not rates: break
        all_rates.extend(rates)
        if len(rates) < 1000: break
        since = rates[-1]['timestamp'] + 1
        time.sleep(0.05)
    df = pd.DataFrame([{
        'ts': pd.Timestamp(r['datetime']).tz_localize(None),
        'fr': r['fundingRate']
    } for r in all_rates])
    df = df.set_index('ts').sort_index()
    # 计算滚动均值
    for w in FR_MA_WINDOWS:
        df[f'fr_ma{w}'] = df['fr'].rolling(w).mean()
    fr_cache[symbol] = df
    print(f'done ({len(df)}条)')
    return df

def compute_er(df, n):
    c = df['close']
    return c.diff(n).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)

def get_val_at_entry(trades, index, series):
    """取入场时刻前一个已知值（merge_asof 向后查找）"""
    entry_df = pd.DataFrame({'ts': trades['entry_dt']})
    ref_df   = series.reset_index()
    ref_df.columns = ['ts', 'val']
    merged = pd.merge_asof(entry_df.sort_values('ts'), ref_df, on='ts', direction='backward')
    merged = merged.set_index(entry_df.sort_values('ts').index)
    return merged['val'].reindex(trades.index).values

# ─── 打印工具 ─────────────────────────────────────────────────────────────────

def lift_str(filtered_pnl, base):
    pct = (filtered_pnl - base) / abs(base) * 100 if base != 0 else 0
    return f'{pct:>+6.1f}%'

def print_table(title, rows_data, base, er_base):
    """rows_data: list of (label, pnl_filtered, keep_rate, er_combo_pnl)"""
    print(f'\n  [{title}]  基准{base:+.0f}  ER≥{ER_TH} {er_base-base:+.0f}({lift_str(er_base, base).strip()})')
    print(f'  {"过滤条件":<28}  {"提升":>8}  {"保留率":>7}  {"ER+组合":>8}')
    print('  ' + '-' * 60)
    for label, pnl_f, keep_rate, er_combo in rows_data:
        print(f'  {label:<28}  {lift_str(pnl_f, base):>8}  {keep_rate*100:>6.0f}%  {lift_str(er_combo, base):>8}')

# ─── 主流程 ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 预拉所有数据
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    print('拉取 K 线数据...')
    for sym in symbols:
        fetch_ohlcv(sym, '2h')
    print('拉取资金费率数据...')
    for sym in symbols:
        fetch_funding_rate(sym)

    all_pnl_list, all_er_list, all_dir_list = [], [], []
    all_fr_raw_list = []
    all_fr_ma = {w: [] for w in FR_MA_WINDOWS}

    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            ohlcv  = fetch_ohlcv(ccxt_sym, '2h')
            fr_df  = fetch_funding_rate(ccxt_sym)

            pnl = trades['pnl'].values * (CAPITAL / 50000)
            er  = compute_er(ohlcv, ER_N)
            er_vals = get_val_at_entry(trades, ohlcv.index, er)

            fr_raw_vals = get_val_at_entry(trades, fr_df.index, fr_df['fr'])
            fr_ma_vals  = {w: get_val_at_entry(trades, fr_df.index, fr_df[f'fr_ma{w}'])
                           for w in FR_MA_WINDOWS}

            all_pnl_list.append(pnl)
            all_er_list.append(er_vals)
            all_dir_list.append(trades['direction'].values)
            all_fr_raw_list.append(fr_raw_vals)
            for w in FR_MA_WINDOWS:
                all_fr_ma[w].append(fr_ma_vals[w])

    all_pnl = np.concatenate(all_pnl_list)
    all_er  = np.concatenate(all_er_list)
    all_dir = np.concatenate(all_dir_list)
    all_fr  = np.concatenate(all_fr_raw_list)
    all_fr_ma_arr = {w: np.concatenate(all_fr_ma[w]) for w in FR_MA_WINDOWS}

    base    = all_pnl.sum()
    er_mask = ~np.isnan(all_er) & (all_er >= ER_TH)
    er_base = all_pnl[er_mask].sum()
    long_m  = all_dir == 'long'
    short_m = all_dir == 'short'

    print(f'\n{"="*72}')
    print(f'v1+v2 合计  {len(all_pnl)}笔  基准{base:+.0f}')
    print(f'多头{long_m.sum()}笔({all_pnl[long_m].sum():+.0f})  空头{short_m.sum()}笔({all_pnl[short_m].sum():+.0f})')
    print(f'{"="*72}')

    # 资金费率分布
    valid_fr = all_fr[~np.isnan(all_fr)]
    print(f'\n资金费率分布（入场时刻）:')
    print(f'  中位数: {np.median(valid_fr)*100:.4f}%  均值: {np.mean(valid_fr)*100:.4f}%')
    print(f'  >0 占比: {(valid_fr>0).mean()*100:.0f}%  <0 占比: {(valid_fr<0).mean()*100:.0f}%')
    for th in NEUTRAL_THS:
        neutral = (np.abs(valid_fr) < th).mean() * 100
        print(f'  |FR| < {th*100:.3f}%: {neutral:.0f}% 的交易')

    # ── 逻辑1：中性过滤（|FR| < 阈值才入场）──
    rows = []
    for w_label, fr_arr in [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS]:
        for th in NEUTRAL_THS:
            mask = ~np.isnan(fr_arr) & (np.abs(fr_arr) < th)
            er_combo = all_pnl[er_mask & mask].sum()
            rows.append((f'|{w_label}| < {th*100:.3f}%', all_pnl[mask].sum(), mask.mean(), er_combo))
    print_table('逻辑1：中性过滤（|FR|<阈值才入场）', rows, base, er_base)

    # ── 逻辑2：方向过滤（做多时FR<阈值，做空时FR>-阈值）──
    rows = []
    for w_label, fr_arr in [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS]:
        for th in NEUTRAL_THS:
            # 做多：费率不能太高（多头拥挤）；做空：费率不能太低（空头拥挤）
            dir_mask = (
                (long_m  & ~np.isnan(fr_arr) & (fr_arr < th)) |
                (short_m & ~np.isnan(fr_arr) & (fr_arr > -th))
            )
            er_combo = all_pnl[er_mask & dir_mask].sum()
            rows.append((f'{w_label} 方向过滤 th={th*100:.3f}%', all_pnl[dir_mask].sum(), dir_mask.mean(), er_combo))
    print_table('逻辑2：方向过滤（避免同向拥挤）', rows, base, er_base)

    # ── 逻辑3：极端反向加分（费率极端反向时允许入场，其余跳过）──
    rows = []
    for w_label, fr_arr in [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS]:
        for th in [0.0003, 0.0005, 0.0008, 0.001]:
            # 做多时费率极端负（空头拥挤，逼空行情）；做空时费率极端正（多头拥挤，逼多行情）
            extreme_mask = (
                (long_m  & ~np.isnan(fr_arr) & (fr_arr < -th)) |
                (short_m & ~np.isnan(fr_arr) & (fr_arr >  th))
            )
            er_combo = all_pnl[er_mask & extreme_mask].sum()
            rows.append((f'{w_label} 极端反向 th={th*100:.3f}%', all_pnl[extreme_mask].sum(), extreme_mask.mean(), er_combo))
    print_table('逻辑3：极端反向（逼空/逼多时入场）', rows, base, er_base)

    # ── 逻辑4：极端同向排除（费率极端同向时不入场）──
    rows = []
    for w_label, fr_arr in [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS]:
        for th in [0.0003, 0.0005, 0.0008, 0.001]:
            # 排除做多时费率极端正（多头拥挤）和做空时费率极端负（空头拥挤）
            exclude_mask = (
                (long_m  & ~np.isnan(fr_arr) & (fr_arr >  th)) |
                (short_m & ~np.isnan(fr_arr) & (fr_arr < -th))
            )
            keep_mask = ~exclude_mask | np.isnan(fr_arr)
            er_combo = all_pnl[er_mask & keep_mask].sum()
            rows.append((f'{w_label} 排除同向 th={th*100:.3f}%', all_pnl[keep_mask].sum(), keep_mask.mean(), er_combo))
    print_table('逻辑4：排除极端同向拥挤交易', rows, base, er_base)

    # ── 汇总：各逻辑最优 ──
    print(f'\n{"="*72}')
    print('各逻辑最优汇总')
    print(f'{"="*72}')
    print(f'  基准:          {base:+.0f}')
    print(f'  ER≥{ER_TH}:       {er_base:+.0f}  ({lift_str(er_base, base).strip()})')

    # 找各逻辑最优
    for logic_name, fr_options, th_options, mask_fn in [
        ('中性过滤', [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS],
         NEUTRAL_THS,
         lambda fr, th: ~np.isnan(fr) & (np.abs(fr) < th)),
        ('方向过滤', [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS],
         NEUTRAL_THS,
         lambda fr, th: (long_m & ~np.isnan(fr) & (fr < th)) | (short_m & ~np.isnan(fr) & (fr > -th))),
        ('极端反向', [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS],
         [0.0003, 0.0005, 0.0008, 0.001],
         lambda fr, th: (long_m & ~np.isnan(fr) & (fr < -th)) | (short_m & ~np.isnan(fr) & (fr > th))),
        ('排除同向', [('原始FR', all_fr)] + [(f'MA{w}', all_fr_ma_arr[w]) for w in FR_MA_WINDOWS],
         [0.0003, 0.0005, 0.0008, 0.001],
         lambda fr, th: ~((long_m & ~np.isnan(fr) & (fr > th)) | (short_m & ~np.isnan(fr) & (fr < -th))) | np.isnan(fr)),
    ]:
        best_pnl, best_label = -np.inf, ''
        for w_label, fr_arr in fr_options:
            for th in th_options:
                mask = mask_fn(fr_arr, th)
                pnl  = all_pnl[mask].sum()
                if pnl > best_pnl:
                    best_pnl, best_label = pnl, f'{w_label} th={th*100:.3f}%'
        print(f'  {logic_name}最优: {best_pnl:+.0f}  ({lift_str(best_pnl, base).strip()})  [{best_label}]')
