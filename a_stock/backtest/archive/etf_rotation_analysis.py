"""
ETF 轮动回测：参数敏感性 + IS/OOS 验证
输出：
  1. 12组参数（Top N × 动量窗口）对比表
  2. 最优参数的 IS/OOS 拆分结果
  3. 净值曲线图（IS/OOS 分段标注）
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非阻塞后端，避免 plt.show() 挂起
import matplotlib.pyplot as plt
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix
from etf_universe import ETF_UNIVERSE

# ── 全局参数 ──────────────────────────────────────────────
INIT_CASH    = 1_000_000
COMMISSION   = 0.0001
SLIPPAGE     = 0.0002
BENCHMARK    = "510300.SH"
START_DATE   = "2016-01-01"
IS_RATIO     = 0.8          # 前80%为样本内

TOP_N_LIST       = [1, 3, 5]
WINDOW_LIST      = [15, 25, 40, 60]

matplotlib.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'Songti SC', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ── 核心函数（复用自 etf_rotation.py）────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix: pd.DataFrame, window: int,
                    risk_adj: bool = False, risk_vol_window: int = 21,
                    skip_days: int = 0) -> pd.DataFrame:
    """
    skip_days > 0：跳过最近 skip_days 日，用 [-window-skip_days, -skip_days) 的价格计算动量，
    规避短期反转干扰（类似学术上的 12-1 动量）。
    """
    scores = {}
    lookback = window + skip_days  # 需要往前取的总天数
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(lookback, len(series)):
            price_window = series.iloc[i - lookback: i - skip_days] if skip_days > 0 else series.iloc[i - window: i]
            raw_score = momentum_score(price_window)
            if risk_adj and i >= risk_vol_window:
                ret_window = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol = ret_window.std() * np.sqrt(252)
                raw_score = raw_score / vol if vol > 1e-6 else raw_score
            score_series.iloc[i] = raw_score
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def get_rebalance_dates_biweekly(index: pd.DatetimeIndex, interval: int = 10) -> list:
    """双周调仓：每隔约 interval 个交易日调仓一次"""
    dates = list(index)
    return [dates[i] for i in range(0, len(dates), interval)]


def run_backtest(close, scores, rebal_dates, top_n, init_cash=INIT_CASH,
                 use_market_filter=False, use_ivol_weighting=False,
                 use_trailing_stop=False, trailing_stop_pct=0.20, cooldown_days=15,
                 use_corr_filter=False, corr_threshold=0.70, corr_window=60,
                 cash_etf=None, breadth_min_pct=0.0):
    MARKET_FILTER_MA = 200
    IVOL_WINDOW = 20
    BENCHMARK_LOCAL = "510300.SH"
    if use_market_filter and BENCHMARK_LOCAL in close.columns:
        benchmark_ma200 = close[BENCHMARK_LOCAL].rolling(MARKET_FILTER_MA).mean()
    else:
        benchmark_ma200 = None

    cash = init_cash
    holdings = {}
    entry_high = {}     # {ts_code: float} 持仓最高价（买入后追踪）
    cooling_down = {}   # {ts_code: pd.Timestamp} 被止损标的的解禁日期
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        # 追踪止损：每日检查持仓最高点回撤
        if use_trailing_stop and holdings:
            for code in list(holdings.keys()):
                price_today = close.loc[date, code] if code in close.columns else None
                if price_today is None or pd.isna(price_today):
                    continue
                entry_high[code] = max(entry_high.get(code, price_today), price_today)
                drawdown = (price_today - entry_high[code]) / entry_high[code]
                if drawdown < -trailing_stop_pct:
                    sell_price = price_today * (1 - SLIPPAGE / 2)
                    cash += holdings[code] * sell_price * (1 - COMMISSION)
                    cooling_down[code] = date + pd.Timedelta(days=cooldown_days)
                    del holdings[code]
                    del entry_high[code]

        if date in rebal_set:
            # 大盘趋势过滤
            if use_market_filter and benchmark_ma200 is not None:
                ma200_val = benchmark_ma200.get(date)
                bench_close = close[BENCHMARK_LOCAL].get(date) if BENCHMARK_LOCAL in close.columns else None
                # MA200 预热期内 rolling mean 为 NaN，视为趋势不满足（保守处理）
                market_in_trend = (
                    pd.notna(bench_close)
                    and pd.notna(ma200_val)
                    and bench_close > ma200_val
                )
            else:
                market_in_trend = True

            if not market_in_trend:
                for code in list(holdings.keys()):
                    price = close.loc[date, code] if code in close.columns else None
                    if price is not None and not pd.isna(price):
                        proceeds = holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                        cash += proceeds
                    del holdings[code]
                    if code in entry_high:
                        del entry_high[code]
                continue

            day_scores = scores.loc[date].dropna()

            # 市场广度过滤：正动量标的占比低于阈值时降为半仓（通过缩减 top_n 实现）
            effective_top_n = top_n
            if breadth_min_pct > 0:
                # 只统计轮动标的池（排除货币ETF）
                pool_scores = day_scores.drop(labels=[cash_etf], errors="ignore") if cash_etf else day_scores
                n_positive = (pool_scores > 0).sum()
                breadth = n_positive / max(len(pool_scores), 1)
                if breadth < breadth_min_pct:
                    effective_top_n = max(1, top_n // 2)  # 降为半仓

            candidate_size = effective_top_n * 3 if use_corr_filter else effective_top_n
            pos_scores = day_scores[day_scores > 0].nlargest(candidate_size)
            # 广度过滤时从正动量池筛选（去除货币ETF）
            if cash_etf:
                pos_scores = pos_scores.drop(labels=[cash_etf], errors="ignore")
            candidates = list(pos_scores.index)

            if use_trailing_stop:
                candidates = [c for c in candidates
                              if c not in cooling_down or cooling_down[c] <= date]

            if use_corr_filter and len(candidates) > 1:
                date_loc = close.index.get_loc(date)
                window_start = max(0, date_loc - corr_window)
                ret_window = close.iloc[window_start:date_loc].pct_change().dropna()
                selected = []
                for code in candidates:
                    if len(selected) >= effective_top_n:
                        break
                    if len(selected) == 0:
                        selected.append(code)
                        continue
                    ok = True
                    for s in selected:
                        pair = ret_window[[code, s]].dropna()
                        if len(pair) < corr_window // 2:
                            continue
                        if pair[code].corr(pair[s]) > corr_threshold:
                            ok = False
                            break
                    if ok:
                        selected.append(code)
                target_codes = selected
            else:
                target_codes = candidates[:effective_top_n]

            for code in list(holdings.keys()):
                if code not in target_codes:
                    price = close.loc[date, code] if code in close.columns else None
                    if price is not None and not pd.isna(price):
                        proceeds = holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                        cash += proceeds
                    del holdings[code]
                    if code in entry_high:
                        del entry_high[code]

            if not target_codes:
                # 无正动量标的：停泊到货币ETF或持现金
                if cash_etf and cash_etf in close.columns:
                    price = close.loc[date, cash_etf]
                    if pd.notna(price) and cash > price:
                        buy_price = price * (1 + SLIPPAGE / 2)
                        buy_shares = int(cash / buy_price / 100) * 100
                        if buy_shares > 0:
                            cost = buy_shares * buy_price * (1 + COMMISSION)
                            if cash >= cost:
                                cash -= cost
                                holdings[cash_etf] = holdings.get(cash_etf, 0) + buy_shares
                continue

            # 仓位分配（等权 or 波动率反比加权）
            # 注意：port_value 是调仓前净值，用于定目标仓位；实际买入受 cash 余额约束
            n = len(target_codes)
            if use_ivol_weighting and n > 0:
                vols = {}
                for code in target_codes:
                    series = close[code].dropna()
                    loc = series.index.get_loc(date) if date in series.index else -1
                    if loc >= IVOL_WINDOW:
                        ret = series.iloc[loc - IVOL_WINDOW: loc].pct_change().dropna()
                        vol = ret.std() * np.sqrt(252)
                        vols[code] = vol if vol > 0 else None
                    else:
                        vols[code] = None
                valid_vols = [v for v in vols.values() if v is not None]
                fallback_vol = float(np.median(valid_vols)) if valid_vols else 1.0
                vols = {c: (v if v is not None else fallback_vol) for c, v in vols.items()}
                inv_vols = {c: 1.0 / v for c, v in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {c: inv_vols[c] / total_inv for c in target_codes}
            else:
                weights = {c: 1.0 / n for c in target_codes}

            for code in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is None or pd.isna(price):
                    continue
                buy_price = price * (1 + SLIPPAGE / 2)
                target_value = port_value * weights[code]
                current_shares = holdings.get(code, 0)
                current_value = current_shares * price
                diff_value = target_value - current_value

                if diff_value > buy_price * 100:
                    buy_shares = int(diff_value / buy_price / 100) * 100
                    if buy_shares > 0:
                        cost = buy_shares * buy_price * (1 + COMMISSION)
                        if cash >= cost:
                            cash -= cost
                            holdings[code] = current_shares + buy_shares
                            if use_trailing_stop and code not in entry_high:
                                entry_high[code] = price
                elif diff_value < -price * 100:
                    sell_shares = int(-diff_value / price / 100) * 100
                    if sell_shares > 0 and current_shares >= sell_shares:
                        proceeds = sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                        cash += proceeds
                        holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


def calc_stats(nav: pd.Series) -> dict:
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    vol = returns.std() * np.sqrt(252)
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Vol": vol,
        "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0,
    }


# ── 加载数据 ──────────────────────────────────────────────

CASH_ETF_CODE = "511880.SH"  # 银华日利（全局）

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = max(WINDOW_LIST) + 20
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]

# 追加货币ETF（不参与动量评分，仅作为空仓停泊标的）
_cash_etf_path = pathlib.Path(__file__).parent.parent.parent / "data" / "daily" / f"{CASH_ETF_CODE}.parquet"
if _cash_etf_path.exists():
    _df = pd.read_parquet(_cash_etf_path, columns=["trade_date", "close"])
    _df["trade_date"] = pd.to_datetime(_df["trade_date"])
    _cash_series = _df.set_index("trade_date")["close"]
    close = close.copy()
    close[CASH_ETF_CODE] = _cash_series.reindex(close.index)
    print(f"货币ETF {CASH_ETF_CODE} 已加载")

print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} → {close.index[-1].date()}")

# IS/OOS 分割点
n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
print(f"IS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} / {split_date.date()} ~ {close.index[-1].date()}")

# 基准净值
bench = close[BENCHMARK].dropna()
bench_nav_full = bench / bench.iloc[0] * INIT_CASH

# ── 参数敏感性：预计算各窗口得分（避免重复计算）────────────

score_cache = {}
for w in WINDOW_LIST:
    print(f"计算动量得分（窗口={w}日）...")
    score_cache[w] = calc_all_scores(close, w)

rebal_dates_full = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]


# ── Part 1: 参数敏感性 ────────────────────────────────────

print("\n运行参数敏感性测试（12组）...")
rows = []
for top_n, window in product(TOP_N_LIST, WINDOW_LIST):
    nav = run_backtest(close, score_cache[window], rebal_dates_full, top_n)
    s = calc_stats(nav)
    rows.append({
        "Top N": top_n,
        "动量窗口": f"{window}日",
        "年化收益": f"{s['CAGR']*100:.1f}%",
        "夏普": f"{s['Sharpe']:.2f}",
        "最大回撤": f"{s['MaxDD']*100:.1f}%",
        "Calmar": f"{s['Calmar']:.2f}",
    })

sensitivity_df = pd.DataFrame(rows)
print("\n" + "=" * 65)
print("参数敏感性（全样本）")
print("=" * 65)
print(sensitivity_df.to_string(index=False))

# 找最优参数（按夏普）
best_row = max(rows, key=lambda x: float(x["夏普"]))
best_top_n = best_row["Top N"]
best_window = int(best_row["动量窗口"].replace("日", ""))
print(f"\n最优参数（夏普最高）：Top{best_top_n}，窗口{best_window}日")


# ── Part 2: IS/OOS 验证（用最优参数）────────────────────

print(f"\n运行 IS/OOS 验证（Top{best_top_n}，窗口{best_window}日）...")

close_is = close[close.index < split_date]
close_oos = close[close.index >= split_date]

scores_best = score_cache[best_window]
scores_is = scores_best[scores_best.index < split_date]
scores_oos = scores_best[scores_best.index >= split_date]

rebal_is = [d for d in rebal_dates_full if d < split_date]
rebal_oos = [d for d in rebal_dates_full if d >= split_date]

nav_is = run_backtest(close_is, scores_is, rebal_is, best_top_n)
nav_oos = run_backtest(close_oos, scores_oos, rebal_oos, best_top_n)

bench_is = bench_nav_full[bench_nav_full.index < split_date]
bench_oos = bench_nav_full[bench_nav_full.index >= split_date]
bench_is = bench_is / bench_is.iloc[0] * INIT_CASH
bench_oos = bench_oos / bench_oos.iloc[0] * INIT_CASH

stats_is = calc_stats(nav_is)
stats_oos = calc_stats(nav_oos)
stats_bench_is = calc_stats(bench_is.reindex(nav_is.index).ffill())
stats_bench_oos = calc_stats(bench_oos.reindex(nav_oos.index).ffill())

print("\n" + "=" * 65)
print(f"IS/OOS 验证结果（Top{best_top_n}，窗口{best_window}日）")
print("=" * 65)

oos_df = pd.DataFrame({
    "指标": ["年化收益(CAGR)", "年化夏普", "最大回撤", "年化波动率", "Calmar"],
    "IS策略": [
        f"{stats_is['CAGR']*100:.1f}%", f"{stats_is['Sharpe']:.2f}",
        f"{stats_is['MaxDD']*100:.1f}%", f"{stats_is['Vol']*100:.1f}%",
        f"{stats_is['Calmar']:.2f}",
    ],
    "OOS策略": [
        f"{stats_oos['CAGR']*100:.1f}%", f"{stats_oos['Sharpe']:.2f}",
        f"{stats_oos['MaxDD']*100:.1f}%", f"{stats_oos['Vol']*100:.1f}%",
        f"{stats_oos['Calmar']:.2f}",
    ],
    "IS基准": [
        f"{stats_bench_is['CAGR']*100:.1f}%", f"{stats_bench_is['Sharpe']:.2f}",
        f"{stats_bench_is['MaxDD']*100:.1f}%", f"{stats_bench_is['Vol']*100:.1f}%",
        f"{stats_bench_is['Calmar']:.2f}",
    ],
    "OOS基准": [
        f"{stats_bench_oos['CAGR']*100:.1f}%", f"{stats_bench_oos['Sharpe']:.2f}",
        f"{stats_bench_oos['MaxDD']*100:.1f}%", f"{stats_bench_oos['Vol']*100:.1f}%",
        f"{stats_bench_oos['Calmar']:.2f}",
    ],
}).set_index("指标")
print(oos_df.to_string())

# 衰减比
sharpe_decay = stats_oos['Sharpe'] / stats_is['Sharpe'] if stats_is['Sharpe'] > 0 else 0
print(f"\nOOS/IS 夏普比：{sharpe_decay:.2f}（>0.5 为可接受）")
if sharpe_decay < 0.5:
    print("警告：OOS 夏普 < IS 夏普 × 0.5，可能存在过拟合")
else:
    print("通过：OOS 表现未显著衰减")


# ── 绘图：全样本净值 + IS/OOS 标注 ───────────────────────

print("\n生成图表...")
nav_full = run_backtest(close, score_cache[best_window], rebal_dates_full, best_top_n)
common = nav_full.index.intersection(bench_nav_full.index)
nav_full = nav_full[common]
bench_aligned = bench_nav_full[common]

fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                          gridspec_kw={'height_ratios': [3, 1, 2]})

# 上图：净值曲线
ax1 = axes[0]
ax1.plot(nav_full.index, nav_full / INIT_CASH, color="#2196F3",
         linewidth=1.5, label=f"ETF轮动 Top{best_top_n} 窗口{best_window}日")
ax1.plot(bench_aligned.index, bench_aligned / INIT_CASH, color="#FF9800",
         linewidth=1.2, alpha=0.8, label="沪深300买持")
ax1.axvline(split_date, color="red", linestyle="--", alpha=0.7, linewidth=1)
ax1.axvspan(close.index[0], split_date, alpha=0.04, color="green")
ax1.axvspan(split_date, close.index[-1], alpha=0.04, color="orange")
ax1.text(close.index[int(split_idx * 0.5)], ax1.get_ylim()[0],
         "样本内(IS)", ha="center", fontsize=9, color="green", alpha=0.8)
ax1.text(close.index[split_idx + int((n_days - split_idx) * 0.5)], ax1.get_ylim()[0],
         "样本外(OOS)", ha="center", fontsize=9, color="darkorange", alpha=0.8)
ax1.set_title(f"ETF趋势轮动 净值曲线（Top{best_top_n}，窗口{best_window}日）")
ax1.set_ylabel("净值")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

# 中图：回撤
ax2 = axes[1]
dd = (nav_full - nav_full.cummax()) / nav_full.cummax() * 100
bench_dd = (bench_aligned - bench_aligned.cummax()) / bench_aligned.cummax() * 100
ax2.fill_between(dd.index, dd, 0, alpha=0.4, color="#2196F3", label="策略回撤")
ax2.fill_between(bench_dd.index, bench_dd, 0, alpha=0.3, color="#FF9800", label="基准回撤")
ax2.axvline(split_date, color="red", linestyle="--", alpha=0.7, linewidth=1)
ax2.set_ylabel("回撤(%)")
ax2.legend(loc="lower left", fontsize=8)
ax2.grid(alpha=0.3)

# 下图：参数敏感性热力图（夏普）
ax3 = axes[2]
heatmap_data = np.zeros((len(TOP_N_LIST), len(WINDOW_LIST)))
for i, top_n in enumerate(TOP_N_LIST):
    for j, window in enumerate(WINDOW_LIST):
        row = next(r for r in rows if r["Top N"] == top_n and r["动量窗口"] == f"{window}日")
        heatmap_data[i, j] = float(row["夏普"])

im = ax3.imshow(heatmap_data, cmap="RdYlGn", aspect="auto",
                vmin=heatmap_data.min() - 0.05, vmax=heatmap_data.max() + 0.05)
ax3.set_xticks(range(len(WINDOW_LIST)))
ax3.set_xticklabels([f"{w}日" for w in WINDOW_LIST])
ax3.set_yticks(range(len(TOP_N_LIST)))
ax3.set_yticklabels([f"Top{n}" for n in TOP_N_LIST])
ax3.set_title("参数敏感性热力图（夏普比率）")
plt.colorbar(im, ax=ax3, orientation="vertical", fraction=0.02)
for i in range(len(TOP_N_LIST)):
    for j in range(len(WINDOW_LIST)):
        ax3.text(j, i, f"{heatmap_data[i,j]:.2f}", ha="center", va="center",
                 fontsize=10, fontweight="bold")

plt.tight_layout()
out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)
fig_path = out_dir / "etf_rotation_analysis.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"图表已保存：{fig_path}")
plt.close('all')

# ── Part 3: 优化对比（使用最优参数）────────────────────────

CASH_ETF = "511880.SH"  # 银华日利，空仓停泊

print(f"\n运行优化对比（Top{best_top_n}，窗口{best_window}日）...")

# 每行：(标签, mf, ivol, ts, corr, cash_etf, risk_adj)
CONFIGS = [
    # (label,                mf,    ivol,  ts,    corr,  cef,       ra)
    ("原始策略",              False, False, False, False, None,      False),
    ("+大盘过滤",             True,  False, False, False, None,      False),
    ("+波动率加权",           False, True,  False, False, None,      False),
    ("+过滤+波动率加权",      True,  True,  False, False, None,      False),
    ("+追踪止损",             False, False, True,  False, None,      False),
    ("+相关性过滤",           False, False, False, True,  None,      False),
    ("+货币ETF空仓",          False, False, False, False, CASH_ETF,  False),
    ("+风险调整动量",         False, False, False, False, None,      True),
    ("+货币ETF+风险调整",     False, True,  False, False, CASH_ETF,  True),
]

scores_opt     = score_cache[best_window]
scores_risk    = calc_all_scores(close, best_window, risk_adj=True)
scores_skip5   = calc_all_scores(close, best_window, skip_days=5)
# IS/OOS 各版本得分
scores_risk_is   = calc_all_scores(close_is,  best_window, risk_adj=True)
scores_risk_oos  = calc_all_scores(close_oos, best_window, risk_adj=True)
scores_skip5_is  = calc_all_scores(close_is,  best_window, skip_days=5)
scores_skip5_oos = calc_all_scores(close_oos, best_window, skip_days=5)

rebal_opt      = rebal_dates_full
rebal_biweekly = get_rebalance_dates_biweekly(close.index, interval=10)
rebal_biweekly_is  = [d for d in rebal_biweekly if d < split_date]
rebal_biweekly_oos = [d for d in rebal_biweekly if d >= split_date]

# CONFIGS 元组格式：(label, mf, ivol, ts, corr, cef, ra, skip5, biweekly, breadth)
# ra=True 用 scores_risk, skip5=True 用 scores_skip5, biweekly=True 用 rebal_biweekly
CONFIGS = [
    # (label,                mf,    ivol,  ts,    corr,  cef,      ra,    sk5,   bwk,   brd)
    ("原始策略",              False, False, False, False, None,     False, False, False, 0.0),
    ("+大盘过滤",             True,  False, False, False, None,     False, False, False, 0.0),
    ("+波动率加权",           False, True,  False, False, None,     False, False, False, 0.0),
    ("+过滤+波动率加权",      True,  True,  False, False, None,     False, False, False, 0.0),
    ("+追踪止损",             False, False, True,  False, None,     False, False, False, 0.0),
    ("+相关性过滤",           False, False, False, True,  None,     False, False, False, 0.0),
    ("+货币ETF空仓",          False, False, False, False, CASH_ETF, False, False, False, 0.0),
    ("+风险调整动量",         False, False, False, False, None,     True,  False, False, 0.0),
    ("+货币ETF+风险调整",     False, True,  False, False, CASH_ETF, True,  False, False, 0.0),
    ("+跳过近5日",            False, False, False, False, None,     False, True,  False, 0.0),
    ("+双周调仓",             False, False, False, False, None,     False, False, True,  0.0),
    ("+广度过滤30%",          False, False, False, False, None,     False, False, False, 0.30),
]

compare_rows = []
compare_navs = {}

for label, use_mf, use_iv, use_ts, use_corr, cef, ra, sk5, bwk, brd in CONFIGS:
    sc = scores_risk if ra else (scores_skip5 if sk5 else scores_opt)
    rb = rebal_biweekly if bwk else rebal_opt
    nav_c = run_backtest(close, sc, rb, best_top_n,
                         use_market_filter=use_mf, use_ivol_weighting=use_iv,
                         use_trailing_stop=use_ts, use_corr_filter=use_corr,
                         cash_etf=cef, breadth_min_pct=brd)
    s = calc_stats(nav_c)
    compare_rows.append({
        "配置":     label,
        "年化收益": f"{s['CAGR']*100:.1f}%",
        "夏普":     f"{s['Sharpe']:.2f}",
        "最大回撤": f"{s['MaxDD']*100:.1f}%",
        "年化波动": f"{s['Vol']*100:.1f}%",
        "Calmar":   f"{s['Calmar']:.2f}",
    })
    compare_navs[label] = nav_c

compare_df = pd.DataFrame(compare_rows).set_index("配置")
print("\n" + "=" * 70)
print("优化对比（全样本）")
print("=" * 70)
print(compare_df.to_string())

# 对比净值曲线
fig2, ax = plt.subplots(figsize=(14, 8))
colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336", "#FF9800",
          "#9C27B0", "#00BCD4", "#795548", "#E91E63", "#607D8B", "#FF5722", "#8BC34A"]
for (label, *_), color in zip(CONFIGS, colors):
    nav_c = compare_navs[label]
    ax.plot(nav_c.index, nav_c / INIT_CASH, label=label, color=color, linewidth=1.4)
ax.plot(bench_nav_full.index, bench_nav_full / INIT_CASH,
        color="#FF9800", linewidth=1.2, alpha=0.7, linestyle="--", label="沪深300买持")
ax.set_title(f"策略优化对比净值曲线（Top{best_top_n}，窗口{best_window}日）")
ax.set_ylabel("净值")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
plt.tight_layout()
fig2_path = out_dir / "etf_rotation_compare.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"\n对比图表已保存：{fig2_path}")
plt.close('all')

# IS/OOS 验证最优配置
best_label = max(compare_navs, key=lambda lbl: calc_stats(compare_navs[lbl])["Sharpe"])
best_cfg = next(cfg for cfg in CONFIGS if cfg[0] == best_label)
_, best_mf, best_iv, best_ts, best_corr, best_cef, best_ra, best_sk5, best_bwk, best_brd = best_cfg

print(f"\n最高夏普配置：{best_label}")
print("运行 IS/OOS 验证...")

sc_is  = scores_risk_is  if best_ra else (scores_skip5_is  if best_sk5 else scores_is)
sc_oos = scores_risk_oos if best_ra else (scores_skip5_oos if best_sk5 else scores_oos)
rb_is  = rebal_biweekly_is  if best_bwk else rebal_is
rb_oos = rebal_biweekly_oos if best_bwk else rebal_oos

nav_best_is  = run_backtest(close_is, sc_is, rb_is, best_top_n,
                             use_market_filter=best_mf, use_ivol_weighting=best_iv,
                             use_trailing_stop=best_ts, use_corr_filter=best_corr,
                             cash_etf=best_cef, breadth_min_pct=best_brd)
nav_best_oos = run_backtest(close_oos, sc_oos, rb_oos, best_top_n,
                             use_market_filter=best_mf, use_ivol_weighting=best_iv,
                             use_trailing_stop=best_ts, use_corr_filter=best_corr,
                             cash_etf=best_cef, breadth_min_pct=best_brd)

s_is  = calc_stats(nav_best_is)
s_oos = calc_stats(nav_best_oos)
print(f"IS  夏普：{s_is['Sharpe']:.2f}，年化收益：{s_is['CAGR']*100:.1f}%，最大回撤：{s_is['MaxDD']*100:.1f}%")
print(f"OOS 夏普：{s_oos['Sharpe']:.2f}，年化收益：{s_oos['CAGR']*100:.1f}%，最大回撤：{s_oos['MaxDD']*100:.1f}%")

decay = s_oos["Sharpe"] / s_is["Sharpe"] if s_is["Sharpe"] > 0 else 0
print(f"OOS/IS 夏普比：{decay:.2f}（>0.5 为可接受）")
if decay < 0.5:
    print("警告：OOS 夏普 < IS 夏普 × 0.5，可能存在过拟合")
else:
    print("通过：OOS 表现未显著衰减")
