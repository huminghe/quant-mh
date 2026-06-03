"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
HMM 制度识别入场过滤验证（2026-06-03）

验证：用隐马尔可夫模型（HMM）识别市场制度（趋势/震荡），
只在"趋势制度"下允许新开仓，是否能改善策略表现。

核心逻辑：
  - 入场信号：沿用原始 TV EMA 策略信号（从 xlsx 读取）
  - HMM 过滤：每次入场前，用滚动窗口训练 2 状态 HMM，
    判断当前市场制度。只在"趋势制度"下允许入场。
  - 制度识别特征：log 收益率 + ATR%（波动率），2 维特征
  - 趋势制度判定：|均值收益率| 更高的状态 = 趋势状态

时区：TV 导出 UTC+8 naive → 减 8 小时 → UTC，与 Binance 对齐

重要：为避免 lookahead bias，HMM 只使用入场时刻之前的数据训练。
每次入场点用过去 N 根 bar 滚动训练，预测当前所处制度。

参数网格：
  - hmm_window（训练窗口）：126, 252, 504（约 6周/3月/6月 的 8H bar）
  - n_states：2, 3（2 状态：趋势/震荡；3 状态：上涨/下跌/震荡）

总组合：2 × 3 = 6 组

用法：
  python hmm_regime_filter_validation.py
  python hmm_regime_filter_validation.py --detail
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path
from hmmlearn import hmm as hmmlib

parser = argparse.ArgumentParser()
parser.add_argument('--detail', action='store_true', help='按标的展示最优参数详细结果')
args = parser.parse_args()

BASE_CAPITAL = 10_000
COMMISSION   = 0.0008

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

# 参数网格
HMM_WINDOWS = [126, 252, 504]
N_STATES_LIST = [2, 3]

# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_trades(fname):
    path = downloads / fname
    if not path.exists():
        print(f'  ⚠ 找不到文件：{fname}')
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
            pnl_key   = '净损益 USDT' if '净损益 USDT' in col_idx else 'Profit USDT'
            price_key = next((k for k in col_idx if '价格' in k or k == 'Price'), None)
            pnl   = row[col_idx.get(pnl_key, -1)]
            price = row[col_idx[price_key]] if price_key else None
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt']    = pd.Timestamp(dt)
                by_num[num]['entry_price'] = float(price) if price is not None else np.nan
                by_num[num]['direction']   = 1 if ('多' in typ or 'Long' in typ) else -1
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['exit_dt']    = pd.Timestamp(dt)
                by_num[num]['exit_price'] = float(price) if price is not None else np.nan
                by_num[num]['pnl']        = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values()
                if all(k in d for k in ('entry_dt','exit_dt','entry_price','exit_price','pnl','direction'))]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    df['exit_dt']  = (pd.to_datetime(df['exit_dt'])  - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol):
    if symbol in ohlcv_cache: return ohlcv_cache[symbol]
    print(f'  拉取 {symbol} 8H...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, '8h', since=since, limit=1000)
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
    ohlcv_cache[symbol] = df
    print('done')
    return df

# ─── HMM 特征预计算 ───────────────────────────────────────────────────────────

def build_features(ohlcv_df, atr_period=14):
    """
    构建 HMM 特征矩阵：
      - log_ret：对数收益率
      - atr_pct：ATR / close（归一化波动率）
    """
    close = ohlcv_df['close']
    high  = ohlcv_df['high']
    low   = ohlcv_df['low']

    log_ret = np.log(close / close.shift(1)).fillna(0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    atr_pct = (atr / close).fillna(0)

    return pd.DataFrame({'log_ret': log_ret, 'atr_pct': atr_pct},
                        index=ohlcv_df.index)

# ─── 制度识别（批量预计算，避免逐笔训练）──────────────────────────────────────

def build_regime_series(features_df, window, n_states, step=20):
    """
    批量对每根 bar 打制度标签，避免逐笔训练。

    策略：
      - 每 step 根 bar 重新训练一次 HMM（滚动窗口）
      - 用训练好的模型对接下来 step 根 bar 预测制度
      - 趋势状态 = |均值 log_ret| 最大的状态（或 3 状态取前两）

    返回：pd.Series，index=features_df.index，值为 True（趋势）/ False（震荡）
    """
    X_all = features_df[['log_ret', 'atr_pct']].values
    n = len(X_all)
    is_trend = np.ones(n, dtype=bool)  # 默认允许（数据不足时）

    i = window
    current_model = None

    while i < n:
        train_X = X_all[i - window: i]

        try:
            model = hmmlib.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=80,
                random_state=42,
            )
            model.fit(train_X)
            current_model = model

            # 趋势状态判定
            means = model.means_[:, 0]
            abs_means = np.abs(means)
            if n_states == 2:
                trend_states = {int(np.argmax(abs_means))}
            else:
                trend_states = set(int(x) for x in np.argsort(abs_means)[-2:])

            # 对接下来 step 根 bar 打标签
            end = min(i + step, n)
            pred_X = X_all[i: end]
            states = model.predict(pred_X)
            for j, s in enumerate(states):
                is_trend[i + j] = (s in trend_states)

        except Exception:
            # 训练失败，保持默认 True
            pass

        i += step

    return pd.Series(is_trend, index=features_df.index)

# ─── 汇总统计 ─────────────────────────────────────────────────────────────────

def calc_stats(pnl_series, n_years=7, n_combos=8):
    total  = pnl_series.sum()
    wr     = (pnl_series > 0).mean() * 100
    wins   = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    rr     = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 0
    cum    = pnl_series.cumsum()
    dd     = ((cum - cum.cummax()) / (BASE_CAPITAL * n_combos) * 100).min()
    sharpe = ((pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series) / n_years)
              if pnl_series.std() > 0 else 0)
    return dict(total=total, wr=wr, rr=rr, max_dd=dd, sharpe=sharpe, n=len(pnl_series))

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run():
    print("=== HMM 制度识别入场过滤验证 ===")
    print("HMM 特征：log_ret + ATR%（双变量高斯 HMM）")
    print("趋势制度判定：|均值收益率| 最大的状态")
    print("批量模式：每20根bar重新训练，对接下来20根打标签（避免逐笔训练）")
    print(f"参数网格：window={HMM_WINDOWS}, n_states={N_STATES_LIST}")
    print(f"总组合：{len(HMM_WINDOWS) * len(N_STATES_LIST)}\n")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("加载交易记录...")
    all_trades = {}
    for ver, sym_files in VERSION_FILES.items():
        for sym, (fname, ccxt_sym) in sym_files.items():
            trades = load_trades(fname)
            if not trades.empty:
                all_trades[(ver, sym)] = (trades, ccxt_sym)

    if not all_trades:
        print("错误：没有找到任何交易记录文件")
        return

    print(f"已加载：{[f'{v}_{s}' for v,s in all_trades]}")

    print("\n拉取 OHLCV 数据...")
    all_ohlcv = {}
    for (ver, sym), (trades, ccxt_sym) in all_trades.items():
        if ccxt_sym not in all_ohlcv:
            all_ohlcv[ccxt_sym] = fetch_ohlcv(ccxt_sym)

    # ── 基准统计 ──────────────────────────────────────────────────────────────
    baseline_pnl = pd.concat([t for t, _ in all_trades.values()])['pnl']
    baseline = calc_stats(baseline_pnl)
    print(f"\n{'─'*72}")
    print(f"基准（无过滤）：Sharpe={baseline['sharpe']:.3f}  "
          f"Total={baseline['total']:+,.0f}  MaxDD={baseline['max_dd']:.1f}%  "
          f"WR={baseline['wr']:.1f}%  RR={baseline['rr']:.2f}  N={baseline['n']}")
    print(f"{'─'*72}")

    # ── 预计算特征 ────────────────────────────────────────────────────────────
    print("\n预计算 HMM 特征序列...")
    features_cache = {}
    for ccxt_sym, ohlcv in all_ohlcv.items():
        features_cache[ccxt_sym] = build_features(ohlcv)
    print("done")

    # ── 预计算各参数组合的制度序列 ────────────────────────────────────────────
    print("\n批量训练 HMM，预计算制度标签...")
    regime_cache = {}  # (ccxt_sym, window, n_states) → pd.Series[bool]
    total_combos = len(HMM_WINDOWS) * len(N_STATES_LIST)
    combo_idx = 0
    for n_states in N_STATES_LIST:
        for window in HMM_WINDOWS:
            combo_idx += 1
            for ccxt_sym, features_df in features_cache.items():
                key = (ccxt_sym, window, n_states)
                if key not in regime_cache:
                    print(f"  [{combo_idx}/{total_combos}] {ccxt_sym} states={n_states} win={window}...",
                          end=' ', flush=True)
                    regime_cache[key] = build_regime_series(features_df, window, n_states)
                    trend_rate = regime_cache[key].mean() * 100
                    print(f"趋势比例 {trend_rate:.1f}%")
    print("done\n")

    # ── 参数扫描 ──────────────────────────────────────────────────────────────
    results = []
    for n_states in N_STATES_LIST:
        for window in HMM_WINDOWS:
            filtered_pnls = []
            skipped_pnls  = []
            n_passed = 0

            for (ver, sym), (trades, ccxt_sym) in all_trades.items():
                regime_s = regime_cache[(ccxt_sym, window, n_states)]

                for _, row in trades.iterrows():
                    # 取入场时刻之前最近一根 bar 的制度标签
                    entry_dt = row['entry_dt']
                    valid = regime_s[regime_s.index <= entry_dt]
                    if valid.empty:
                        is_trend = True  # 数据不足，不过滤
                    else:
                        is_trend = bool(valid.iloc[-1])

                    if is_trend:
                        filtered_pnls.append(row['pnl'])
                        n_passed += 1
                    else:
                        skipped_pnls.append(row['pnl'])

            pass_rate = n_passed / baseline['n'] * 100
            st       = calc_stats(pd.Series(filtered_pnls)) if filtered_pnls else {}
            skip_st  = calc_stats(pd.Series(skipped_pnls)) if skipped_pnls else {}

            sharpe_diff = st.get('sharpe', 0) - baseline['sharpe']
            print(f"  states={n_states}, window={window}: "
                  f"Sharpe {st.get('sharpe',0):.3f} ({sharpe_diff:+.3f}) "
                  f"通过率 {pass_rate:.1f}%  "
                  f"被过滤Sharpe {skip_st.get('sharpe',0):.3f}")

            results.append({
                'n_states':    n_states,
                'window':      window,
                'sharpe':      round(st.get('sharpe', 0), 3),
                'total':       round(st.get('total', 0), 0),
                'max_dd':      round(st.get('max_dd', 0), 1),
                'wr':          round(st.get('wr', 0), 1),
                'rr':          round(st.get('rr', 0), 2),
                'pass%':       round(pass_rate, 1),
                'skip_sharpe': round(skip_st.get('sharpe', 0), 3),
                'sharpe_diff': round(sharpe_diff, 3),
            })

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"结果汇总（按 Sharpe 排序，基准={baseline['sharpe']:.3f}）")
    print(f"{'─'*80}")

    best  = results_df.iloc[0]
    worst = results_df.iloc[-1]

    print(f"最好：Sharpe {best['sharpe']:.3f} ({best['sharpe_diff']:+.3f}) "
          f"| states={int(best['n_states'])}, window={int(best['window'])} "
          f"| 通过率 {best['pass%']:.1f}%")
    print(f"最差：Sharpe {worst['sharpe']:.3f} ({worst['sharpe_diff']:+.3f}) "
          f"| states={int(worst['n_states'])}, window={int(worst['window'])} "
          f"| 通过率 {worst['pass%']:.1f}%")

    n_better = (results_df['sharpe_diff'] > 0).sum()
    print(f"\n优于基准的组合：{n_better}/{len(results_df)}")

    print(f"\n全部结果（按 Sharpe 排序）:")
    header = (f"{'states':>7} {'window':>7} {'sharpe':>7} {'diff':>6} "
              f"{'total':>9} {'maxDD%':>7} {'WR%':>5} {'RR':>5} "
              f"{'pass%':>6} {'skip_sh':>8}")
    print(header)
    print('─' * 68)
    for _, r in results_df.iterrows():
        sign = '+' if r['sharpe_diff'] >= 0 else ''
        print(f"{int(r['n_states']):>7} {int(r['window']):>7} "
              f"{r['sharpe']:>7.3f} {sign}{r['sharpe_diff']:>5.3f} "
              f"{r['total']:>+9,.0f} {r['max_dd']:>7.1f} "
              f"{r['wr']:>5.1f} {r['rr']:>5.2f} "
              f"{r['pass%']:>6.1f} {r['skip_sharpe']:>8.3f}")

    # ── 诊断：被过滤的交易质量 ────────────────────────────────────────────────
    print(f"\n诊断：skip_sharpe > 基准 = 过滤器在扔掉好交易（结构性失效特征）")
    any_warn = False
    for _, r in results_df.iterrows():
        if r['skip_sharpe'] > baseline['sharpe']:
            print(f"  ⚠ states={int(r['n_states'])}, window={int(r['window'])}: "
                  f"被过滤交易 Sharpe={r['skip_sharpe']:.3f} > 基准 {baseline['sharpe']:.3f}")
            any_warn = True
    if not any_warn:
        print("  （无异常——过滤掉的交易质量均不高于基准）")

    # ── 详细标的视图 ──────────────────────────────────────────────────────────
    if args.detail:
        best_row = results_df.iloc[0]
        bn = int(best_row['n_states'])
        bw = int(best_row['window'])

        print(f"\n{'─'*72}")
        print(f"最优参数按标的拆分（states={bn}, window={bw}）")
        print(f"{'─'*72}")

        for (ver, sym), (trades, ccxt_sym) in sorted(all_trades.items()):
            regime_s = regime_cache[(ccxt_sym, bw, bn)]
            passed, skipped = [], []
            for _, row in trades.iterrows():
                valid = regime_s[regime_s.index <= row['entry_dt']]
                is_trend = bool(valid.iloc[-1]) if not valid.empty else True
                (passed if is_trend else skipped).append(row['pnl'])

            orig_s = calc_stats(trades['pnl'], n_combos=1)
            pass_s = calc_stats(pd.Series(passed), n_combos=1) if passed else {}
            diff   = pass_s.get('sharpe', 0) - orig_s['sharpe']
            sign   = '+' if diff >= 0 else ''
            pass_rate = len(passed) / len(trades) * 100
            print(f"  {ver}_{sym:<5} | 原 Sharpe {orig_s['sharpe']:5.3f} → "
                  f"{pass_s.get('sharpe',0):5.3f} ({sign}{diff:.3f}) | "
                  f"通过率 {pass_rate:.1f}% ({len(passed)}/{len(trades)})")

    print(f"\n总交易笔数（基准）：{baseline['n']}")
    print(f"参数总组合：{len(results_df)}")
    print(f"\n结论参考：若优于基准的组合 < 20%，HMM 制度过滤方向可视为无效。")

if __name__ == '__main__':
    run()
