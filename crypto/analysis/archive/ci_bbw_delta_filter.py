"""
CI/BBW 变化率入场过滤验证（2026-05-29）

核心假设：
  之前测试的是 CI/BBW 绝对值（CI ≤ 55 等），全部无效。
  本次测试"状态转换"信号：
    - CI 下降：ci[0] < ci[1]（从混乱转趋势的时刻）
    - CI 从高位下降：ci[1] > 阈值 且 ci[0] < ci[1]（更强的转换信号）
    - BBW 上升：bbw[0] > bbw[1]（波动率从压缩开始释放）
    - BBW 连续上升：bbw[0] > bbw[1] > bbw[2]
    - BBW 低位扩张：bbw[0] > bbw[1] 且 bbw[1] < 30th percentile

时区修正：TV 时间 -8h 转 UTC

用法：
  python /tmp/ci_bbw_delta_filter.py
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

def calc_ci(df, n):
    """Choppiness Index CI(n)，值域 [0, 100]，越高越混乱"""
    atr1 = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_sum = atr1.rolling(n).sum()
    hh = df['high'].rolling(n).max()
    ll = df['low'].rolling(n).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100 * np.log10(atr_sum / rng) / np.log10(n)

def calc_bbw(df, n=20, mult=2.0):
    """Bollinger Band Width = (上轨 - 下轨) / 中轨"""
    mid  = df['close'].rolling(n).mean()
    std  = df['close'].rolling(n).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return (upper - lower) / mid.replace(0, np.nan)

def get_vals_at_entry(trades_df, ohlcv_df, *series_list):
    """
    对多个 series，同时取每笔交易入场时刻之前最近已收盘 K 线的值。
    返回 list of arrays，与 series_list 顺序对应。
    """
    # 构建合并 df
    ind_df = pd.DataFrame({'ts': ohlcv_df.index})
    for i, s in enumerate(series_list):
        ind_df[f'v{i}'] = s.values
    ind_df['ts'] = ind_df['ts'].astype('datetime64[ns]')

    trades = trades_df.copy()
    trades['entry_dt'] = trades['entry_dt'].astype('datetime64[ns]')

    merged = pd.merge_asof(
        trades.sort_values('entry_dt'),
        ind_df.sort_values('ts'),
        left_on='entry_dt', right_on='ts',
        direction='backward'
    )
    return [merged[f'v{i}'].values for i in range(len(series_list))]

# ─── 过滤器评估 ───────────────────────────────────────────────────────────────

def eval_filter(all_pnl, mask, base, label):
    """计算过滤后 PnL 变化"""
    if mask.sum() == 0:
        return {'label': label, 'pct': np.nan, 'keep': 0}
    filt_pnl = all_pnl[mask].sum()
    pct = (filt_pnl - base) / abs(base) * 100
    keep = mask.sum() / len(all_pnl) * 100
    return {'label': label, 'pct': pct, 'keep': keep}

# ─── 主验证 ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 72)
    print("CI/BBW 变化率入场过滤验证（时区修正版）")
    print("=" * 72)

    # 加载交易数据
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

        # 收集各 (ver, sym) 的数据
        sym_data = {}
        for (ver, sym, ccxt_sym), trades_df in all_trades.items():
            ohlcv = fetch_ohlcv(ccxt_sym, tf)
            pnl = trades_df['pnl'].values * (CAPITAL / 50000)

            results = {}
            for ci_n in [7, 10, 14]:
                ci = calc_ci(ohlcv, ci_n)
                ci_prev = ci.shift(1)
                ci_prev2 = ci.shift(2)
                vals = get_vals_at_entry(trades_df, ohlcv, ci, ci_prev, ci_prev2)
                results[f'ci{ci_n}']      = vals[0]   # ci[0]
                results[f'ci{ci_n}_p1']   = vals[1]   # ci[1]
                results[f'ci{ci_n}_p2']   = vals[2]   # ci[2]

            for bbw_n in [14, 20]:
                bbw = calc_bbw(ohlcv, bbw_n)
                bbw_prev = bbw.shift(1)
                bbw_prev2 = bbw.shift(2)
                # 30th percentile（滚动 252 根 bar）
                bbw_pct30 = bbw.rolling(252, min_periods=50).quantile(0.30)
                vals = get_vals_at_entry(trades_df, ohlcv, bbw, bbw_prev, bbw_prev2, bbw_pct30)
                results[f'bbw{bbw_n}']      = vals[0]
                results[f'bbw{bbw_n}_p1']   = vals[1]
                results[f'bbw{bbw_n}_p2']   = vals[2]
                results[f'bbw{bbw_n}_pct30'] = vals[3]

            sym_data[(ver, sym)] = {'pnl': pnl, **results}

        # 合并所有标的
        all_pnl = np.concatenate([d['pnl'] for d in sym_data.values()])
        base = all_pnl.sum()
        print(f"\n基准 PnL：{base:+.0f}，总交易笔数：{len(all_pnl)}\n")

        results_list = []

        # ── CI 变化率测试 ──────────────────────────────────────────────────────
        print("  [CI 变化率]")
        for ci_n in [7, 10, 14]:
            ci0  = np.concatenate([d[f'ci{ci_n}']    for d in sym_data.values()])
            ci1  = np.concatenate([d[f'ci{ci_n}_p1'] for d in sym_data.values()])
            ci2  = np.concatenate([d[f'ci{ci_n}_p2'] for d in sym_data.values()])

            # 变体1：CI 下降（ci[0] < ci[1]）
            mask = ~np.isnan(ci0) & ~np.isnan(ci1) & (ci0 < ci1)
            r = eval_filter(all_pnl, mask, base, f'CI({ci_n}) 下降')
            results_list.append(r)
            print(f"    CI({ci_n}) 下降:              {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

            # 变体2：CI 从高位（>55）开始下降
            for hi_th in [55, 61.8]:
                mask = ~np.isnan(ci0) & ~np.isnan(ci1) & (ci0 < ci1) & (ci1 > hi_th)
                r = eval_filter(all_pnl, mask, base, f'CI({ci_n}) 从>{hi_th:.0f}下降')
                results_list.append(r)
                print(f"    CI({ci_n}) 从>{hi_th:.0f}下降:       {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

            # 变体3：CI 连续下降（ci[0] < ci[1] < ci[2]）
            mask = (~np.isnan(ci0) & ~np.isnan(ci1) & ~np.isnan(ci2)
                    & (ci0 < ci1) & (ci1 < ci2))
            r = eval_filter(all_pnl, mask, base, f'CI({ci_n}) 连续下降')
            results_list.append(r)
            print(f"    CI({ci_n}) 连续下降:          {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

        # ── BBW 变化率测试 ─────────────────────────────────────────────────────
        print("\n  [BBW 变化率]")
        for bbw_n in [14, 20]:
            bbw0   = np.concatenate([d[f'bbw{bbw_n}']       for d in sym_data.values()])
            bbw1   = np.concatenate([d[f'bbw{bbw_n}_p1']    for d in sym_data.values()])
            bbw2   = np.concatenate([d[f'bbw{bbw_n}_p2']    for d in sym_data.values()])
            pct30  = np.concatenate([d[f'bbw{bbw_n}_pct30'] for d in sym_data.values()])

            # 变体1：BBW 上升（bbw[0] > bbw[1]）
            mask = ~np.isnan(bbw0) & ~np.isnan(bbw1) & (bbw0 > bbw1)
            r = eval_filter(all_pnl, mask, base, f'BBW({bbw_n}) 上升')
            results_list.append(r)
            print(f"    BBW({bbw_n}) 上升:             {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

            # 变体2：BBW 连续上升（bbw[0] > bbw[1] > bbw[2]）
            mask = (~np.isnan(bbw0) & ~np.isnan(bbw1) & ~np.isnan(bbw2)
                    & (bbw0 > bbw1) & (bbw1 > bbw2))
            r = eval_filter(all_pnl, mask, base, f'BBW({bbw_n}) 连续上升')
            results_list.append(r)
            print(f"    BBW({bbw_n}) 连续上升:         {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

            # 变体3：BBW 从低位（<30th pct）开始上升
            mask = (~np.isnan(bbw0) & ~np.isnan(bbw1) & ~np.isnan(pct30)
                    & (bbw0 > bbw1) & (bbw1 < pct30))
            r = eval_filter(all_pnl, mask, base, f'BBW({bbw_n}) 低位扩张')
            results_list.append(r)
            print(f"    BBW({bbw_n}) 低位扩张:         {r['pct']:>+6.1f}%  保留{r['keep']:.0f}%")

        # 汇总最好结果
        valid = [r for r in results_list if not np.isnan(r['pct'])]
        if valid:
            best = max(valid, key=lambda x: x['pct'])
            worst = min(valid, key=lambda x: x['pct'])
            print(f"\n  最好：{best['label']}  {best['pct']:+.1f}%（保留{best['keep']:.0f}%）")
            print(f"  最差：{worst['label']}  {worst['pct']:+.1f}%（保留{worst['keep']:.0f}%）")

    print("\n" + "=" * 72)
    print("完成")

if __name__ == '__main__':
    run()
