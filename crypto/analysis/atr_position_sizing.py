"""
ATR 动态定仓验证 v2（2026-05-29）

背景：
  当前策略固定仓位约 10000 USDT/笔，最大回撤约 15%。
  目标：ATR 动态定仓，平均仓位保持在 10000 USDT 附近，高波动减仓，低波动加仓。

两种定仓方法对比：

  方法 A — ATR 归一化（推荐）：
    atr_pct = ATR(n) / close                          # 当前波动率（百分比）
    median_atr_pct = atr_pct 的历史中位数（滚动窗口）
    allocation = BASE * (median_atr_pct / atr_pct)    # 高波动减仓，低波动加仓
    allocation = clip(allocation, BASE/cap, BASE*cap)  # cap 控制上下限倍数
    → 平均仓位天然接近 BASE，无需调参

  方法 B — Vol Targeting（CTA 标准）：
    annual_vol = ATR(n) / close * sqrt(365 * bars_per_day)
    allocation = BASE * (TARGET_VOL / annual_vol)
    allocation = clip(allocation, MIN_ALLOC, MAX_ALLOC)
    → 需要手动设置 TARGET_VOL 使平均仓位接近 BASE

时区处理：
  TV 导出时间是 UTC+8 naive datetime，减 8 小时转 UTC，再与 Binance UTC 数据 merge_asof。

用法：
  python atr_position_sizing.py              # 默认参数扫描
  python atr_position_sizing.py --detail     # 输出各标的详细结果
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--detail', action='store_true', help='输出各标的详细结果')
args = parser.parse_args()

BASE_CAPITAL = 10_000   # 目标平均仓位（USDT），与实盘一致

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

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易记录，TV UTC+8 时间 -8h 转 UTC"""
    path = downloads / fname
    if not path.exists():
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
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
    key = (symbol, tf)
    if key in ohlcv_cache: return ohlcv_cache[key]
    print(f'  拉取 {symbol} {tf}...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                break
            except Exception as e:
                if attempt == 2: raise
                import time; time.sleep(3)
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

# ─── ATR 计算 ─────────────────────────────────────────────────────────────────

def calc_atr(df, n):
    """ATR(n)，Wilder 平滑"""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ─── 方法 A：ATR 归一化 ───────────────────────────────────────────────────────

def alloc_atr_normalized(ohlcv_df, atr_n, median_window, cap_mult):
    """
    allocation = BASE * (median_atr_pct / current_atr_pct)
    median_atr_pct：过去 median_window 根 bar 的 ATR% 中位数
    cap_mult：上下限倍数，allocation ∈ [BASE/cap_mult, BASE*cap_mult]
    """
    atr = calc_atr(ohlcv_df, atr_n)
    atr_pct = atr / ohlcv_df['close']
    median_atr = atr_pct.rolling(median_window, min_periods=atr_n).median()
    raw = BASE_CAPITAL * (median_atr / atr_pct)
    return raw.clip(BASE_CAPITAL / cap_mult, BASE_CAPITAL * cap_mult)

# ─── 方法 B：Vol Targeting ────────────────────────────────────────────────────

def alloc_vol_targeting(ohlcv_df, atr_n, tf, target_vol, cap_mult):
    """
    allocation = BASE * (target_vol / annual_vol)
    annual_vol = ATR(n)/close * sqrt(365 * bars_per_day)
    """
    bars_per_day = {'1h': 24, '2h': 12, '4h': 6, '8h': 3, '1d': 1}.get(tf, 12)
    annual_factor = np.sqrt(365 * bars_per_day)
    atr = calc_atr(ohlcv_df, atr_n)
    annual_vol = (atr / ohlcv_df['close']) * annual_factor
    raw = BASE_CAPITAL * (target_vol / annual_vol)
    return raw.clip(BASE_CAPITAL / cap_mult, BASE_CAPITAL * cap_mult)

# ─── 评估 ─────────────────────────────────────────────────────────────────────

def evaluate(trades_df, alloc_series):
    """用 allocation 缩放每笔 PnL，返回 merged DataFrame"""
    alloc_df = alloc_series.rename('alloc').reset_index()
    alloc_df.columns = ['ts', 'alloc']
    alloc_df['ts'] = alloc_df['ts'].astype('datetime64[ns]')
    trades = trades_df.copy()
    trades['entry_dt'] = trades['entry_dt'].astype('datetime64[ns]')
    merged = pd.merge_asof(
        trades.sort_values('entry_dt'),
        alloc_df.sort_values('ts'),
        left_on='entry_dt', right_on='ts',
        direction='backward'
    )
    merged['pnl_atr'] = merged['pnl'] * (merged['alloc'] / BASE_CAPITAL)
    return merged

def stats(pnl_series, n_years=7):
    """计算统计指标，回撤分母用 BASE_CAPITAL * 8（8个策略×标的组合）"""
    total = pnl_series.sum()
    win_rate = (pnl_series > 0).mean() * 100
    wins   = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
    cum = pnl_series.cumsum()
    dd = ((cum - cum.cummax()) / (BASE_CAPITAL * 8) * 100).min()
    n = len(pnl_series)
    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(n / n_years) if pnl_series.std() > 0 else 0
    return dict(total=total, win_rate=win_rate, rr=rr, max_dd=dd, sharpe=sharpe,
                mean_alloc=0, pct_at_min=0, pct_at_max=0)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== ATR 动态定仓验证 v2 ===")
    print(f"基准仓位：{BASE_CAPITAL} USDT/笔（与实盘一致）")
    print("时区处理：TV UTC+8 -8h → UTC，与 Binance UTC 数据对齐\n")

    # 加载交易数据
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df
    print(f"已加载 {len(all_trades)} 个策略×标的组合\n")

    # 拉取所有需要的 OHLCV（每个 symbol×tf 只拉一次）
    tfs_needed = ['1h', '2h', '4h', '8h', '1d']
    symbols = set(ccxt_sym for _, _, ccxt_sym in all_trades.keys())
    for sym in symbols:
        for tf in tfs_needed:
            fetch_ohlcv(sym, tf)
    print()

    # 固定仓位基准
    fixed_pnl_all = []
    for trades_df in all_trades.values():
        fixed_pnl_all.extend(trades_df['pnl'].values)
    fixed_pnl_all = pd.Series(fixed_pnl_all)
    s_fixed = stats(fixed_pnl_all)
    print(f"固定仓位基准（{BASE_CAPITAL} USDT）：")
    print(f"  总PnL={s_fixed['total']:+.0f}  胜率={s_fixed['win_rate']:.1f}%  "
          f"盈亏比={s_fixed['rr']:.2f}  最大回撤={s_fixed['max_dd']:.1f}%  Sharpe≈{s_fixed['sharpe']:.2f}\n")

    # ── 方法 A 扫描：ATR 归一化 ──────────────────────────────────────────────
    print("─── 方法 A：ATR 归一化（median_atr / current_atr × BASE）───")
    print(f"  {'时间框架':>6s}  {'ATR周期':>6s}  {'中位数窗口':>8s}  {'上下限倍数':>8s}  "
          f"{'均值仓位':>8s}  {'总PnL':>8s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs基准':>7s}")

    best_a = None
    for tf in ['1h', '2h', '4h', '8h', '1d']:
        for atr_n in [10, 14, 20, 30]:
            for med_win in [100, 200, 500]:
                for cap in [2.0, 3.0, 4.0]:
                    pnl_list, alloc_list = [], []
                    for (ver, sym, ccxt_sym), trades_df in all_trades.items():
                        ohlcv = ohlcv_cache[(ccxt_sym, tf)]
                        alloc = alloc_atr_normalized(ohlcv, atr_n, med_win, cap)
                        merged = evaluate(trades_df, alloc)
                        pnl_list.extend(merged['pnl_atr'].values)
                        alloc_list.extend(merged['alloc'].values)
                    s = stats(pd.Series(pnl_list))
                    mean_alloc = np.mean(alloc_list)
                    vs = (pd.Series(pnl_list).sum() - s_fixed['total']) / abs(s_fixed['total']) * 100
                    min_alloc = BASE_CAPITAL / cap
                    max_alloc = BASE_CAPITAL * cap
                    pct_min = (np.array(alloc_list) <= min_alloc * 1.01).mean() * 100
                    pct_max = (np.array(alloc_list) >= max_alloc * 0.99).mean() * 100
                    row = dict(tf=tf, atr_n=atr_n, med_win=med_win, cap=cap,
                               mean_alloc=mean_alloc, total=pd.Series(pnl_list).sum(),
                               max_dd=s['max_dd'], sharpe=s['sharpe'], vs=vs,
                               pct_min=pct_min, pct_max=pct_max)
                    if best_a is None or s['sharpe'] > best_a['sharpe']:
                        best_a = row
                    print(f"  {tf:>6s}  {atr_n:>6d}  {med_win:>8d}  {cap:>8.1f}x  "
                          f"{mean_alloc:>7.0f}  {pd.Series(pnl_list).sum():>+8.0f}  "
                          f"{s['max_dd']:>7.1f}%  {s['sharpe']:>6.2f}  {vs:>+6.1f}%"
                          f"  [触底{pct_min:.0f}% 触顶{pct_max:.0f}%]")

    print(f"\n  方法A最优（Sharpe最高）：{best_a['tf']} ATR({best_a['atr_n']}) "
          f"中位数窗口={best_a['med_win']} 上下限={best_a['cap']:.1f}x  "
          f"均值仓位={best_a['mean_alloc']:.0f} 最大回撤={best_a['max_dd']:.1f}% Sharpe={best_a['sharpe']:.2f}")

    # ── 方法 B 扫描：Vol Targeting ────────────────────────────────────────────
    print("\n─── 方法 B：Vol Targeting（BASE × target_vol / annual_vol）───")
    print(f"  {'时间框架':>6s}  {'ATR周期':>6s}  {'目标波动率':>8s}  {'上下限倍数':>8s}  "
          f"{'均值仓位':>8s}  {'总PnL':>8s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs基准':>7s}")

    best_b = None
    # 先探测各标的实际年化波动率，确定合理的 target_vol 范围
    # 加密货币年化波动率约 60-120%，target_vol 设在这个范围才能让平均仓位接近 BASE
    for tf in ['1h', '2h', '4h', '8h', '1d']:
        bars_per_day = {'1h': 24, '2h': 12, '4h': 6, '8h': 3, '1d': 1}[tf]
        annual_factor = np.sqrt(365 * bars_per_day)
        for atr_n in [10, 14, 20, 30]:
            for target_vol in [0.50, 0.70, 0.90, 1.10]:
                for cap in [2.0, 3.0, 4.0]:
                    pnl_list, alloc_list = [], []
                    for (ver, sym, ccxt_sym), trades_df in all_trades.items():
                        ohlcv = ohlcv_cache[(ccxt_sym, tf)]
                        alloc = alloc_vol_targeting(ohlcv, atr_n, tf, target_vol, cap)
                        merged = evaluate(trades_df, alloc)
                        pnl_list.extend(merged['pnl_atr'].values)
                        alloc_list.extend(merged['alloc'].values)
                    s = stats(pd.Series(pnl_list))
                    mean_alloc = np.mean(alloc_list)
                    vs = (pd.Series(pnl_list).sum() - s_fixed['total']) / abs(s_fixed['total']) * 100
                    min_alloc = BASE_CAPITAL / cap
                    max_alloc = BASE_CAPITAL * cap
                    pct_min = (np.array(alloc_list) <= min_alloc * 1.01).mean() * 100
                    pct_max = (np.array(alloc_list) >= max_alloc * 0.99).mean() * 100
                    row = dict(tf=tf, atr_n=atr_n, target_vol=target_vol, cap=cap,
                               mean_alloc=mean_alloc, total=pd.Series(pnl_list).sum(),
                               max_dd=s['max_dd'], sharpe=s['sharpe'], vs=vs,
                               pct_min=pct_min, pct_max=pct_max)
                    if best_b is None or s['sharpe'] > best_b['sharpe']:
                        best_b = row
                    print(f"  {tf:>6s}  {atr_n:>6d}  {target_vol:>8.0%}  {cap:>8.1f}x  "
                          f"{mean_alloc:>7.0f}  {pd.Series(pnl_list).sum():>+8.0f}  "
                          f"{s['max_dd']:>7.1f}%  {s['sharpe']:>6.2f}  {vs:>+6.1f}%"
                          f"  [触底{pct_min:.0f}% 触顶{pct_max:.0f}%]")

    print(f"\n  方法B最优（Sharpe最高）：{best_b['tf']} ATR({best_b['atr_n']}) "
          f"目标波动率={best_b['target_vol']:.0%} 上下限={best_b['cap']:.1f}x  "
          f"均值仓位={best_b['mean_alloc']:.0f} 最大回撤={best_b['max_dd']:.1f}% Sharpe={best_b['sharpe']:.2f}")

    # ── 汇总对比 ──────────────────────────────────────────────────────────────
    print("\n─── 最终对比 ───")
    print(f"  {'方案':30s}  {'均值仓位':>8s}  {'总PnL':>8s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs基准':>7s}")
    print(f"  {'固定仓位（基准）':30s}  {BASE_CAPITAL:>7.0f}  {s_fixed['total']:>+8.0f}  "
          f"{s_fixed['max_dd']:>7.1f}%  {s_fixed['sharpe']:>6.2f}  {'—':>7s}")

    for label, best in [('方法A最优', best_a), ('方法B最优', best_b)]:
        desc = (f"{label}({best.get('tf','?')} ATR{best.get('atr_n','?')})" if label else '')
        print(f"  {desc:30s}  {best['mean_alloc']:>7.0f}  {best['total']:>+8.0f}  "
              f"{best['max_dd']:>7.1f}%  {best['sharpe']:>6.2f}  {best['vs']:>+6.1f}%")

    print(f"\n注：最大回撤分母 = BASE_CAPITAL × 8 = {BASE_CAPITAL*8} USDT（8个策略×标的组合）")
    print("Sharpe 为粗略估计（假设 7 年数据，交易笔数均匀分布）")

    # ── 各标的详细结果（--detail 模式）────────────────────────────────────────
    if args.detail:
        # ── 仓位分布分析（方法A最优参数）────────────────────────────────────────
        print("\n\n═══ 仓位分布分析（方法A最优参数：2H ATR30, 窗口=500, cap=4x） ═══")
        all_allocs = []
        for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
            ohlcv = ohlcv_cache[(ccxt_sym, best_a['tf'])]
            alloc = alloc_atr_normalized(ohlcv, best_a['atr_n'], best_a['med_win'], best_a['cap'])
            merged = evaluate(trades_df, alloc)
            all_allocs.extend(merged['alloc'].values)

        allocs = np.array(all_allocs)
        buckets = [
            (0,     2500,  "< 2,500  (< 0.25x)"),
            (2500,  5000,  "2,500–5,000  (0.25–0.5x)"),
            (5000,  7500,  "5,000–7,500  (0.5–0.75x)"),
            (7500,  10000, "7,500–10,000  (0.75–1x)"),
            (10000, 12500, "10,000–12,500  (1–1.25x)"),
            (12500, 15000, "12,500–15,000  (1.25–1.5x)"),
            (15000, 20000, "15,000–20,000  (1.5–2x)"),
            (20000, 30000, "20,000–30,000  (2–3x)"),
            (30000, 99999, "> 30,000  (> 3x)"),
        ]
        total_n = len(allocs)
        print(f"\n  总入场笔数：{total_n}，基准仓位：{BASE_CAPITAL:,} USDT")
        print(f"  均值：{allocs.mean():.0f}  中位数：{np.median(allocs):.0f}  "
              f"最小：{allocs.min():.0f}  最大：{allocs.max():.0f}\n")
        print(f"  {'仓位区间':30s}  {'笔数':>6s}  {'占比':>6s}  {'累计':>6s}  图示")
        cum = 0
        for lo, hi, label in buckets:
            n = ((allocs >= lo) & (allocs < hi)).sum()
            pct = n / total_n * 100
            cum += pct
            bar = '█' * int(pct / 2)
            print(f"  {label:30s}  {n:>6d}  {pct:>5.1f}%  {cum:>5.1f}%  {bar}")

        # 各标的单独分布
        print(f"\n  各标的均值仓位：")
        print(f"  {'策略':>4s}  {'标的':>5s}  {'均值':>7s}  {'中位数':>7s}  {'p10':>7s}  {'p25':>7s}  {'p75':>7s}  {'p90':>7s}")
        for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
            ohlcv = ohlcv_cache[(ccxt_sym, best_a['tf'])]
            alloc = alloc_atr_normalized(ohlcv, best_a['atr_n'], best_a['med_win'], best_a['cap'])
            merged = evaluate(trades_df, alloc)
            a = merged['alloc'].values
            print(f"  {ver:>4s}  {sym:>5s}  {a.mean():>7.0f}  {np.median(a):>7.0f}  "
                  f"{np.percentile(a,10):>7.0f}  {np.percentile(a,25):>7.0f}  "
                  f"{np.percentile(a,75):>7.0f}  {np.percentile(a,90):>7.0f}")

        print("\n\n═══ 各策略×标的 详细结果（方法A最优参数） ═══")
        print(f"参数：{best_a['tf']} ATR({best_a['atr_n']}) 中位数窗口={best_a['med_win']} 上下限={best_a['cap']:.1f}x\n")
        print(f"  {'策略':>4s}  {'标的':>5s}  {'笔数':>5s}  {'固定PnL':>9s}  {'ATR PnL':>9s}  "
              f"{'均值仓位':>8s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs固定':>7s}")
        print("  " + "─" * 80)

        for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
            ohlcv = ohlcv_cache[(ccxt_sym, best_a['tf'])]
            alloc = alloc_atr_normalized(ohlcv, best_a['atr_n'], best_a['med_win'], best_a['cap'])
            merged = evaluate(trades_df, alloc)

            fixed_pnl = trades_df['pnl']
            atr_pnl   = merged['pnl_atr']
            n_trades  = len(fixed_pnl)

            # 单标的回撤分母用 BASE_CAPITAL（单策略×标的）
            def single_stats(pnl_s, n_years=7):
                total = pnl_s.sum()
                cum = pnl_s.cumsum()
                dd = ((cum - cum.cummax()) / BASE_CAPITAL * 100).min()
                sharpe = (pnl_s.mean() / pnl_s.std()) * np.sqrt(len(pnl_s) / n_years) if pnl_s.std() > 0 else 0
                return total, dd, sharpe

            f_total, f_dd, f_sharpe = single_stats(fixed_pnl)
            a_total, a_dd, a_sharpe = single_stats(atr_pnl)
            mean_alloc = merged['alloc'].mean()
            vs = (a_total - f_total) / abs(f_total) * 100 if f_total != 0 else 0

            print(f"  {ver:>4s}  {sym:>5s}  {n_trades:>5d}  {f_total:>+9.0f}  {a_total:>+9.0f}  "
                  f"{mean_alloc:>7.0f}  {a_dd:>7.1f}%  {a_sharpe:>6.2f}  {vs:>+6.1f}%")

        print("\n  注：单标的最大回撤分母 = BASE_CAPITAL = 10000 USDT")

        # 方法B最优参数也输出一遍
        print(f"\n\n═══ 各策略×标的 详细结果（方法B最优参数） ═══")
        print(f"参数：{best_b['tf']} ATR({best_b['atr_n']}) 目标波动率={best_b['target_vol']:.0%} 上下限={best_b['cap']:.1f}x\n")
        print(f"  {'策略':>4s}  {'标的':>5s}  {'笔数':>5s}  {'固定PnL':>9s}  {'ATR PnL':>9s}  "
              f"{'均值仓位':>8s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs固定':>7s}")
        print("  " + "─" * 80)

        for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
            ohlcv = ohlcv_cache[(ccxt_sym, best_b['tf'])]
            alloc = alloc_vol_targeting(ohlcv, best_b['atr_n'], best_b['tf'], best_b['target_vol'], best_b['cap'])
            merged = evaluate(trades_df, alloc)

            fixed_pnl = trades_df['pnl']
            atr_pnl   = merged['pnl_atr']
            n_trades  = len(fixed_pnl)

            f_total, f_dd, f_sharpe = single_stats(fixed_pnl)
            a_total, a_dd, a_sharpe = single_stats(atr_pnl)
            mean_alloc = merged['alloc'].mean()
            vs = (a_total - f_total) / abs(f_total) * 100 if f_total != 0 else 0

            print(f"  {ver:>4s}  {sym:>5s}  {n_trades:>5d}  {f_total:>+9.0f}  {a_total:>+9.0f}  "
                  f"{mean_alloc:>7.0f}  {a_dd:>7.1f}%  {a_sharpe:>6.2f}  {vs:>+6.1f}%")

        print("\n  注：单标的最大回撤分母 = BASE_CAPITAL = 10000 USDT")

        # ── 各标的 × 各时间周期 最优参数对比（方法A）────────────────────────────
        print("\n\n═══ 各标的 × 各时间周期 最优 Sharpe（方法A，ATR归一化） ═══")
        print("每格：最优参数 | ATR PnL | 最大回撤 | Sharpe | vs固定\n")

        tfs = ['1h', '2h', '4h', '8h', '1d']
        a_params = [(atr_n, med_win, cap)
                    for atr_n in [10, 14, 20, 30]
                    for med_win in [100, 200, 500]
                    for cap in [2.0, 3.0, 4.0]]

        for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
            fixed_pnl = trades_df['pnl']
            f_total = fixed_pnl.sum()
            print(f"  ── {ver} {sym}（固定PnL={f_total:+.0f}）──")
            print(f"  {'TF':>4s}  {'最优参数':^22s}  {'ATR PnL':>9s}  {'最大回撤':>8s}  {'Sharpe':>7s}  {'vs固定':>7s}")

            for tf in tfs:
                ohlcv = ohlcv_cache[(ccxt_sym, tf)]
                best = None
                for atr_n, med_win, cap in a_params:
                    alloc = alloc_atr_normalized(ohlcv, atr_n, med_win, cap)
                    merged = evaluate(trades_df, alloc)
                    pnl_s = merged['pnl_atr']
                    total = pnl_s.sum()
                    cum = pnl_s.cumsum()
                    dd = ((cum - cum.cummax()) / BASE_CAPITAL * 100).min()
                    sharpe = (pnl_s.mean() / pnl_s.std()) * np.sqrt(len(pnl_s) / 7) if pnl_s.std() > 0 else 0
                    vs = (total - f_total) / abs(f_total) * 100 if f_total != 0 else 0
                    if best is None or sharpe > best['sharpe']:
                        best = dict(atr_n=atr_n, med_win=med_win, cap=cap,
                                    total=total, dd=dd, sharpe=sharpe, vs=vs)
                param_str = f"ATR{best['atr_n']} w={best['med_win']} c={best['cap']:.0f}x"
                print(f"  {tf:>4s}  {param_str:^22s}  {best['total']:>+9.0f}  "
                      f"{best['dd']:>7.1f}%  {best['sharpe']:>6.2f}  {best['vs']:>+6.1f}%")
            print()

if __name__ == '__main__':
    run()
