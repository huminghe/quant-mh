"""
Kelly 准则动态定仓验证（2026-06-03）

研究问题：
  用滚动窗口估计胜率和盈亏比，按 Kelly 比例动态调整仓位，是否优于固定仓位或 ATR 定仓？

Kelly 公式：
  f* = W - (1-W)/B
    W = 滚动窗口内胜率
    B = 滚动窗口内平均盈利 / 平均亏损（盈亏比）
  f* 可为负（表示当前没有统计优势），此时仓位设为最小值

仓位归一化：
  allocation = BASE * (f* / mean_f*)    # 相对于历史均值 Kelly 的倍数
  clip(allocation, BASE/cap, BASE*cap)  # 上下限保护

参数扫描：
  滚动窗口：20 / 50 / 100 笔
  Kelly 分数：0.25x / 0.5x / 1.0x（全 Kelly 风险太大，常用分数）
  Cap：2x / 3x / 4x

对比基准：
  1. 固定仓位（10000 USDT）
  2. ATR 归一化最优参数（2H ATR30，窗口=500，cap=4x）—— 上次验证结论

数据：BTC/ETH/SOL/DOGE，v1+v2，2019-2026
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

BASE_CAPITAL = 10_000
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
    """加载交易记录，TV UTC+8 -8h 转 UTC"""
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
            pnl_key = next((k for k in col_idx if '净损益' in k and 'USDT' in k), None) or \
                      next((k for k in col_idx if 'Profit' in k and 'USDT' in k), None) or \
                      '净损益 USDT'
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
    if not rows_out:
        return pd.DataFrame()
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol, tf='2h'):
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

# ─── ATR 归一化定仓（基准对比用，上次验证最优参数）────────────────────────────

def calc_atr(df, n=30):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def alloc_atr_normalized(ohlcv_df, atr_n=30, median_window=500, cap_mult=4.0):
    atr = calc_atr(ohlcv_df, atr_n)
    atr_pct = atr / ohlcv_df['close']
    median_atr = atr_pct.rolling(median_window, min_periods=atr_n).median()
    raw = BASE_CAPITAL * (median_atr / atr_pct)
    return raw.clip(BASE_CAPITAL / cap_mult, BASE_CAPITAL * cap_mult)

# ─── Kelly 定仓 ───────────────────────────────────────────────────────────────

def calc_kelly_alloc(trades_df, window, kelly_frac, cap_mult):
    """
    对每笔交易，用之前 window 笔交易的胜率和盈亏比估算 Kelly f*，
    按 fraction×f* 归一化后得到仓位大小。

    注意：第 0~window-1 笔交易因样本不足，仓位设为 BASE（固定仓位）。
    """
    pnl = trades_df['pnl'].values
    n = len(pnl)
    alloc = np.full(n, BASE_CAPITAL, dtype=float)

    for i in range(window, n):
        window_pnl = pnl[i-window:i]
        wins = window_pnl[window_pnl > 0]
        losses = window_pnl[window_pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            # 没有亏损或全亏，无法估计 Kelly，用固定仓位
            alloc[i] = BASE_CAPITAL
            continue
        W = len(wins) / window          # 胜率
        B = wins.mean() / abs(losses.mean())   # 盈亏比
        f_star = W - (1 - W) / B       # Kelly 比例
        alloc[i] = f_star               # 先存原始值

    # 归一化：用有效 f* 值的均值作为分母，确保均值仓位≈BASE
    valid_mask = np.arange(n) >= window
    raw_f = alloc.copy()
    valid_f = raw_f[valid_mask]
    # 只用正 f* 归一化（负 f* 表示当前无优势，设为最小仓位）
    pos_f = valid_f[valid_f > 0]
    if len(pos_f) == 0:
        # 全部无优势，回退到固定仓位
        return pd.Series(np.full(n, BASE_CAPITAL), index=trades_df.index)
    mean_f = pos_f.mean()

    # 将 f* 转换为仓位
    for i in range(window, n):
        f = raw_f[i]
        if f <= 0:
            # 无统计优势，使用最小仓位
            alloc[i] = BASE_CAPITAL / cap_mult
        else:
            alloc[i] = BASE_CAPITAL * kelly_frac * (f / mean_f)
    alloc = np.clip(alloc, BASE_CAPITAL / cap_mult, BASE_CAPITAL * cap_mult)
    return pd.Series(alloc, index=trades_df.index)

# ─── 统计指标 ─────────────────────────────────────────────────────────────────

def stats(pnl_series, n_years=7, n_strats=8):
    total = pnl_series.sum()
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    win_rate = len(wins) / len(pnl_series) * 100 if len(pnl_series) > 0 else 0
    rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
    cum = pnl_series.cumsum()
    dd = ((cum - cum.cummax()) / (BASE_CAPITAL * n_strats) * 100).min()
    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series) / n_years) \
             if pnl_series.std() > 0 else 0
    return dict(total=total, win_rate=win_rate, rr=rr, max_dd=dd, sharpe=sharpe)

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== Kelly 准则动态定仓验证 ===")
    print(f"基准仓位：{BASE_CAPITAL} USDT/笔")
    print("对比方案：固定仓位 / ATR归一化最优（2H ATR30 窗口500 cap=4x）/ Kelly 各参数组合\n")

    # 加载交易数据
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            df = load_trades(fname)
            if not df.empty:
                all_trades[(ver, sym, ccxt_sym)] = df
                print(f"  已加载 {ver} {sym}：{len(df)} 笔交易")
    print(f"\n共 {len(all_trades)} 个策略×标的组合\n")

    # 拉取 ATR 对比用的 OHLCV
    symbols = set(ccxt_sym for _, _, ccxt_sym in all_trades.keys())
    for sym in symbols:
        fetch_ohlcv(sym, '2h')
    print()

    # ── 基准1：固定仓位 ────────────────────────────────────────────────────────
    fixed_pnl_all = pd.Series(
        [p for df in all_trades.values() for p in df['pnl'].values]
    )
    s_fixed = stats(fixed_pnl_all)
    print(f"基准1 固定仓位（{BASE_CAPITAL} USDT）：")
    print(f"  总PnL={s_fixed['total']:+.0f}  胜率={s_fixed['win_rate']:.1f}%  "
          f"盈亏比={s_fixed['rr']:.2f}  最大回撤={s_fixed['max_dd']:.1f}%  Sharpe={s_fixed['sharpe']:.3f}\n")

    # ── 基准2：ATR 归一化最优 ──────────────────────────────────────────────────
    atr_pnl_all = []
    for (ver, sym, ccxt_sym), trades_df in all_trades.items():
        ohlcv = ohlcv_cache[(ccxt_sym, '2h')]
        alloc = alloc_atr_normalized(ohlcv)
        alloc_df = alloc.rename('alloc').reset_index()
        alloc_df.columns = ['ts', 'alloc']
        alloc_df['ts'] = alloc_df['ts'].astype('datetime64[ns]')
        merged = pd.merge_asof(
            trades_df.sort_values('entry_dt'),
            alloc_df.sort_values('ts'),
            left_on='entry_dt', right_on='ts', direction='backward'
        )
        merged['pnl_scaled'] = merged['pnl'] * (merged['alloc'] / BASE_CAPITAL)
        atr_pnl_all.extend(merged['pnl_scaled'].values)
    atr_pnl_all = pd.Series(atr_pnl_all)
    s_atr = stats(atr_pnl_all)
    print(f"基准2 ATR归一化（2H ATR30 窗口500 cap=4x）：")
    print(f"  总PnL={s_atr['total']:+.0f}  胜率={s_atr['win_rate']:.1f}%  "
          f"盈亏比={s_atr['rr']:.2f}  最大回撤={s_atr['max_dd']:.1f}%  Sharpe={s_atr['sharpe']:.3f}\n")

    # ── Kelly 参数扫描 ─────────────────────────────────────────────────────────
    print("─── Kelly 参数扫描 ───")
    print(f"{'窗口':>6s}  {'Kelly分数':>8s}  {'Cap':>5s}  "
          f"{'均值仓位':>8s}  {'总PnL':>9s}  {'最大回撤':>8s}  {'Sharpe':>7s}  "
          f"{'vs固定':>7s}  {'vs ATR':>7s}")
    print("─" * 85)

    results = []
    for window in [20, 50, 100]:
        for kelly_frac in [0.25, 0.5, 1.0]:
            for cap in [2.0, 3.0, 4.0]:
                pnl_all, alloc_all = [], []
                for (ver, sym, ccxt_sym), trades_df in all_trades.items():
                    alloc = calc_kelly_alloc(trades_df, window, kelly_frac, cap)
                    scaled_pnl = trades_df['pnl'].values * (alloc.values / BASE_CAPITAL)
                    pnl_all.extend(scaled_pnl)
                    alloc_all.extend(alloc.values)
                pnl_s = pd.Series(pnl_all)
                s = stats(pnl_s)
                mean_alloc = np.mean(alloc_all)
                vs_fixed = (pnl_s.sum() - s_fixed['total']) / abs(s_fixed['total']) * 100
                vs_atr = s['sharpe'] - s_atr['sharpe']
                results.append(dict(window=window, kelly_frac=kelly_frac, cap=cap,
                                    mean_alloc=mean_alloc, **s,
                                    vs_fixed=vs_fixed, vs_atr_sharpe=vs_atr))
                print(f"{window:>6d}  {kelly_frac:>8.2f}x  {cap:>4.1f}x  "
                      f"{mean_alloc:>7.0f}  {pnl_s.sum():>+9.0f}  {s['max_dd']:>7.1f}%  "
                      f"{s['sharpe']:>6.3f}  {vs_fixed:>+6.1f}%  {vs_atr:>+7.3f}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['sharpe'].idxmax()]
    worst = results_df.loc[results_df['sharpe'].idxmin()]
    n_beat_fixed = (results_df['sharpe'] > s_fixed['sharpe']).sum()
    n_beat_atr = (results_df['sharpe'] > s_atr['sharpe']).sum()
    total = len(results_df)

    print(f"\n─── 汇总 ───")
    print(f"测试组合数：{total}")
    print(f"优于固定仓位（Sharpe）：{n_beat_fixed}/{total}")
    print(f"优于 ATR 定仓（Sharpe）：{n_beat_atr}/{total}")
    print(f"最好 Sharpe：{best['sharpe']:.3f}（窗口={best['window']:.0f}，"
          f"kelly={best['kelly_frac']:.2f}x，cap={best['cap']:.1f}x）"
          f"  vs固定={best['vs_fixed']:+.1f}%  vs ATR Sharpe差={best['vs_atr_sharpe']:+.3f}")
    print(f"最差 Sharpe：{worst['sharpe']:.3f}（窗口={worst['window']:.0f}，"
          f"kelly={worst['kelly_frac']:.2f}x，cap={worst['cap']:.1f}x）")

    print(f"\n─── 三方案对比 ───")
    print(f"{'方案':32s}  {'均值仓位':>8s}  {'总PnL':>9s}  {'最大回撤':>8s}  {'Sharpe':>7s}")
    print(f"{'固定仓位（基准）':32s}  {BASE_CAPITAL:>7.0f}  "
          f"{s_fixed['total']:>+9.0f}  {s_fixed['max_dd']:>7.1f}%  {s_fixed['sharpe']:>6.3f}")
    print(f"{'ATR 归一化（2H ATR30 窗口500 cap4x）':32s}  {atr_pnl_all.mean():>7.0f}  "
          f"{s_atr['total']:>+9.0f}  {s_atr['max_dd']:>7.1f}%  {s_atr['sharpe']:>6.3f}")
    print(f"{'Kelly 最优':32s}  {best['mean_alloc']:>7.0f}  "
          f"{best['total']:>+9.0f}  {best['max_dd']:>7.1f}%  {best['sharpe']:>6.3f}")

    # ── 诊断：Kelly f* 的时序稳定性 ──────────────────────────────────────────
    print(f"\n─── 诊断：Kelly f* 统计（窗口=50，用于判断信号噪声） ───")
    print(f"{'策略':>4s}  {'标的':>5s}  {'笔数':>5s}  {'f*均值':>7s}  "
          f"{'f*标准差':>8s}  {'f*>0比例':>9s}  {'胜率':>6s}  {'盈亏比':>6s}")
    for (ver, sym, ccxt_sym), trades_df in sorted(all_trades.items()):
        pnl = trades_df['pnl'].values
        n = len(pnl)
        f_stars = []
        for i in range(50, n):
            w_pnl = pnl[i-50:i]
            wins = w_pnl[w_pnl > 0]
            losses = w_pnl[w_pnl < 0]
            if len(wins) == 0 or len(losses) == 0:
                continue
            W = len(wins) / 50
            B = wins.mean() / abs(losses.mean())
            f_stars.append(W - (1 - W) / B)
        if not f_stars:
            continue
        f_arr = np.array(f_stars)
        # 全局胜率/盈亏比
        all_wins = pnl[pnl > 0]
        all_losses = pnl[pnl < 0]
        wr = len(all_wins)/n*100
        rr = abs(all_wins.mean()/all_losses.mean()) if len(all_losses)>0 else 0
        print(f"  {ver:>4s}  {sym:>5s}  {n:>5d}  {f_arr.mean():>7.3f}  "
              f"{f_arr.std():>8.3f}  {(f_arr>0).mean()*100:>8.1f}%  "
              f"{wr:>5.1f}%  {rr:>6.2f}")

    print("\n注：f* 标准差高 → Kelly 信号噪声大，定仓变化接近随机")
    print("注：Sharpe 为粗略估计（假设 7 年数据）")

if __name__ == '__main__':
    run()
