"""
ACF（自相关系数）入场过滤验证 — 时区修正版（2026-05-29）

原脚本（acf_filter_validation.py）存在时区错误：
  TV 导出时间是 UTC+8，Binance API 是 UTC，直接 merge 等效于用了未来 8 小时数据。
本脚本修正时区后重新验证。

测试范围：
  - 版本：v1 + v2，4 标的（BTC/ETH/SOL/DOGE）
  - ACF 时间框架：1h / 2h / 4h / 8h / 1d
  - lag：1 / 3 / 5
  - 阈值：≥0.05 / ≥0.10 / ≥0.15 / ≥0.20
  - 额外测试：ACF 变化率（acf[0] > acf[1]，即 ACF 正在上升）

用法：
  python /tmp/acf_filter_tz_corrected.py
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

CAPITAL    = 20000
ACF_WINDOW = 20
ACF_LAGS   = [1, 3, 5]
ACF_THS    = [0.05, 0.10, 0.15, 0.20]
TEST_TFS   = ['1h', '2h', '4h', '8h', '1d']

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，TV UTC+8 时间 -8h 转 UTC"""
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
            num = row[col_idx['交易 #']]
            typ = str(row[col_idx['类型']])
            dt  = row[col_idx['日期和时间']]
            pnl = row[col_idx['净损益 USDT']]
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
    # 关键：TV 导出时间是 UTC+8，减 8 小时转为 UTC
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

def get_indicator_at_entry(trades_df, ohlcv_df, indicator_series):
    """
    取每笔交易入场时刻之前最近已收盘 K 线的指标值（[1]，不用当前未收盘 K 线）
    时区已对齐：trades entry_dt 是 UTC，ohlcv index 是 UTC
    """
    ind_df = indicator_series.reset_index()
    ind_df.columns = ['ts', 'val']
    ind_df['ts'] = ind_df['ts'].astype('datetime64[ns]')
    trades = trades_df.copy()
    trades['entry_dt'] = trades['entry_dt'].astype('datetime64[ns]')
    merged = pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind_df.sort_values('ts'),
        left_on='entry_dt', right_on='ts',
        direction='backward'
    )
    return merged['val'].values

# ─── 主验证 ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("ACF 入场过滤验证 — 时区修正版（TV -8h → UTC）")
    print("=" * 72)

    # 加载所有交易
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df

    print(f"\n已加载 {len(all_trades)} 个版本×标的组合\n")

    # 合并 v1+v2 所有标的的 pnl（基准）
    all_pnl_list = [df['pnl'].values * (CAPITAL / 50000)
                    for df in all_trades.values()]
    base_total = sum(p.sum() for p in all_pnl_list)
    print(f"基准总 PnL（v1+v2，4标的）：{base_total:+.0f}\n")

    # 按时间框架测试
    for tf in TEST_TFS:
        print(f"\n{'─'*60}")
        print(f"ACF 时间框架：{tf}")
        print(f"{'─'*60}")

        # 收集每个 (ver, sym) 的 ACF 值和 pnl
        sym_data = {}
        for (ver, sym, ccxt_sym), trades_df in all_trades.items():
            ohlcv = fetch_ohlcv(ccxt_sym, tf)
            pnl = trades_df['pnl'].values * (CAPITAL / 50000)
            acf_by_lag = {}
            for lag in ACF_LAGS:
                acf_series = compute_acf(ohlcv, lag)
                acf_vals = get_indicator_at_entry(trades_df, ohlcv, acf_series)
                acf_by_lag[lag] = acf_vals
            # 额外：ACF(lag=1) 变化率（当前 > 上一根，即 ACF 正在上升）
            # 用 lag=1 的 ACF 序列，计算 diff > 0
            acf1 = compute_acf(ohlcv, 1)
            acf1_rising = (acf1.diff() > 0).astype(float).where(acf1.notna(), np.nan)
            acf_by_lag['rising'] = get_indicator_at_entry(trades_df, ohlcv, acf1_rising)
            sym_data[(ver, sym)] = {'pnl': pnl, 'acf': acf_by_lag}

        # 合并所有标的
        all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
        base = all_pnl.sum()

        # 打印表头
        th_header = '  '.join(f'≥{th:.2f}' for th in ACF_THS)
        print(f"\n  {'lag':>8}  {th_header}  rising")
        print(f"  {'':>8}  " + "  ".join(['------'] * len(ACF_THS)) + "  ------")

        for lag in ACF_LAGS:
            acf = np.concatenate([d['acf'][lag] for d in sym_data.values()])
            row = f"  lag={lag:>4}  "
            for th in ACF_THS:
                mask = ~np.isnan(acf) & (acf >= th)
                if mask.sum() == 0:
                    row += " N/A    "
                    continue
                filt_pnl = all_pnl[mask].sum()
                pct = (filt_pnl - base) / abs(base) * 100
                keep = mask.sum() / len(all_pnl) * 100
                row += f"{pct:+.1f}%({keep:.0f}%)  "
            print(row)

        # ACF rising（lag=1 上升）
        rising = np.concatenate([d['acf']['rising'] for d in sym_data.values()])
        mask_r = ~np.isnan(rising) & (rising > 0)
        if mask_r.sum() > 0:
            filt_pnl_r = all_pnl[mask_r].sum()
            pct_r = (filt_pnl_r - base) / abs(base) * 100
            keep_r = mask_r.sum() / len(all_pnl) * 100
            print(f"  {'rising':>8}  {'':>{8*len(ACF_THS)}}  {pct_r:+.1f}%({keep_r:.0f}%)")

        print(f"\n  基准 PnL（{tf}）：{base:+.0f}")

    print("\n" + "=" * 72)
    print("完成")

if __name__ == '__main__':
    run()
