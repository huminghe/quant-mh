"""
第十轮补充调研：用当前线上实际评分公式（风险调整动量）重跑 Top N × 窗口网格，
并直接对这 12 个候选做 PBO。

背景：etf_rotation_analysis.py 的原始 12 组网格（TOP_N_LIST × WINDOW_LIST）
选参数时用的是未风险调整的动量分数，风险调整只在选出 best_window 之后才
追加应用于单点参数，从未针对"风险调整动量"这个真正上线的评分公式重新做过
独立的网格搜索。本脚本补上这一步：

1. 用 risk_adj=True（与 signal_today.py / etf_rotation_v9_qdii_ic.py 完全一致的
   评分公式：OLS斜率×R²÷近21日年化波动率）重新计算 12 组网格；
2. 确认 Top3+窗口25 在真实评分公式下是否仍是全样本夏普最优；
3. 保存每组净值序列，对这 12 候选（同一次网格搜索、同一个调参决策）直接
   计算 PBO——这是对"当前上线策略核心参数"本身的过拟合检验，
   区别于第十轮已做的波动率目标/拥挤度过滤网格（那两个网格不含当前
   上线的核心决策）。
"""

import sys
import pathlib
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix

SKILL_DIR = pathlib.Path.home() / ".claude/plugins/marketplaces/agiprolabs-claude-trading-skills/skills/walk-forward-validation/scripts"
sys.path.insert(0, str(SKILL_DIR))
from overfit_detector import probability_of_backtest_overfitting

# ── 参数（与线上一致）────────────────────────────────────────
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
START_DATE       = "2016-01-01"
RISK_VOL_WINDOW  = 21

TOP_N_LIST  = [1, 3, 5]
WINDOW_LIST = [15, 25, 40, 60]


# ── 信号计算（与线上一致，含风险调整）──────────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_scores_risk_adj(close_matrix: pd.DataFrame, window: int) -> pd.DataFrame:
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            raw = momentum_score(series.iloc[i - window: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close_matrix.index)


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
                  top_n: int, init_cash: float = INIT_CASH) -> pd.Series:
    cash = init_cash
    holdings: dict[str, float] = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        if date not in rebal_set:
            continue

        day_scores = scores.loc[date].dropna()
        pos_scores = day_scores[day_scores > 0].nlargest(top_n * 3)
        target_codes = list(pos_scores.index)[:top_n]

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        n = len(target_codes)
        weights = {c: 1.0 / n for c in target_codes}

        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            buy_price = price * (1 + SLIPPAGE / 2)
            target_value = port_value * weights[code]
            current_shares = holdings.get(code, 0)
            current_value = current_shares * price
            diff = target_value - current_value

            if diff > buy_price * 100:
                buy_shares = int(diff / buy_price / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * buy_price * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = current_shares + buy_shares
            elif diff < -price * 100:
                sell_shares = int(-diff / price / 100) * 100
                if sell_shares > 0 and current_shares >= sell_shares:
                    cash += sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


def calc_stats(nav: pd.Series) -> dict:
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd,
            "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0}


# ── 加载数据（含 QDII，与线上一致）──────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = max(WINDOW_LIST) + 20
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# ── 12 组网格：风险调整动量评分 ──────────────────────────────

print(f"\n计算风险调整动量得分（{len(WINDOW_LIST)} 个窗口）...")
score_cache = {}
for w in WINDOW_LIST:
    score_cache[w] = calc_scores_risk_adj(close, w)

print(f"\n运行风险调整动量参数网格（{len(TOP_N_LIST)}×{len(WINDOW_LIST)}={len(TOP_N_LIST)*len(WINDOW_LIST)}组）...")
navs = {}
rows = []
for top_n, window in product(TOP_N_LIST, WINDOW_LIST):
    nav = run_backtest(close, score_cache[window], rebal_dates, top_n)
    s = calc_stats(nav)
    label = f"top{top_n}_win{window}"
    navs[label] = nav
    rows.append({"Top N": top_n, "窗口": f"{window}日",
                 "年化收益": f"{s['CAGR']*100:.1f}%", "夏普": f"{s['Sharpe']:.3f}",
                 "最大回撤": f"{s['MaxDD']*100:.1f}%", "Calmar": f"{s['Calmar']:.2f}",
                 "_sharpe": s["Sharpe"]})

grid_df = pd.DataFrame(rows)
print("\n" + "=" * 70)
print("风险调整动量参数网格（全样本，含QDII）")
print("=" * 70)
print(grid_df.drop(columns="_sharpe").to_string(index=False))

best_row = grid_df.loc[grid_df["_sharpe"].idxmax()]
print(f"\n最优参数（风险调整动量评分下，夏普最高）：Top{best_row['Top N']}，窗口{best_row['窗口']}")
print(f"当前上线参数：Top3，窗口25日 —— "
      f"{'一致' if (best_row['Top N'] == 3 and best_row['窗口'] == '25日') else '不一致，需关注'}")

# ── 对 12 候选（同一次网格搜索）直接做 PBO ───────────────────

print(f"\n{'=' * 70}\nPBO 检验：12 候选风险调整动量网格（同一次参数搜索，同一个决策）\n{'=' * 70}")

common_index = None
for nav in navs.values():
    idx = nav.dropna().index
    common_index = idx if common_index is None else common_index.intersection(idx)

rets = {label: nav.reindex(common_index).ffill().pct_change() for label, nav in navs.items()}
ret_df = pd.DataFrame(rets).iloc[1:].dropna(axis=1)
print(f"收益矩阵：{ret_df.shape[0]} 个观测 × {ret_df.shape[1]} 个候选")

result = probability_of_backtest_overfitting(ret_df.values, n_groups=6, n_test_groups=2)
print(f"\nPBO = {result.pbo:.3f}（{result.n_overfit_paths}/{result.n_paths}路径过拟合）")
print(f"平均OOS排名 = {result.mean_oos_rank:.3f}  is_overfit = {result.is_overfit}")

is_best_label = ret_df.mean().idxmax()
print(f"\n全样本均值最高的候选：{is_best_label}")
print(f"当前上线参数（top3_win25）在全样本均值排名："
      f"{(ret_df.mean() > ret_df.mean()['top3_win25']).sum() + 1} / {ret_df.shape[1]}")

# 保存净值序列，供追溯
out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
nav_df = pd.DataFrame(navs)
nav_df.to_csv(out_dir / "v12_riskadj_grid_navs.csv")
print(f"\n净值序列已保存：{out_dir / 'v12_riskadj_grid_navs.csv'}")

print("\n完成。")
