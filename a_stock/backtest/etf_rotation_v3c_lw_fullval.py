"""
Ledoit-Wolf 最小方差权重方向完整验证脚本（2026-07）

背景：脏数据（复权bug修复前）判定该方向"负效果"（夏普0.67<基线0.83），
      干净数据重新核实后夏普1.13>基线1.05，出现方向性反转，故做完整稳健性验证。

验证内容（参照 etf_rotation_v3b_fullval.py 拥挤度验证方法论）：
  1. 参数网格扫描（lw_min_history × 协方差历史窗口），统计超基线比例
  2. 逐年净值对比（检查改善是否被单一年份/行情驱动）
  3. 滚动3年窗口夏普稳健性
  4. 最优参数 IS/OOS 验证
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
BENCHMARK        = "510300.SH"
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21

# 网格：lw_min_history × 协方差历史窗口长度
MIN_HISTORY_GRID = [30, 60, 90, 120]
HIST_WINDOW_GRID = [126, 189, 252]

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 信号计算（与 v3 一致）─────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score(series.iloc[i - MOMENTUM_WINDOW: i])
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


# ── 回测主逻辑（可配置 lw_min_history / hist_window）────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    use_ledoit_wolf: bool = False,
    lw_min_history: int = 60,
    hist_window: int = 252,
) -> pd.Series:
    cash = init_cash
    holdings = {}
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
        candidates = list(pos_scores.index)
        target_codes = candidates[:top_n]

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        n = len(target_codes)
        if use_ledoit_wolf and n >= 2:
            date_loc = close.index.get_loc(date)
            hist_start = max(0, date_loc - hist_window)
            ret_hist = close[target_codes].iloc[hist_start:date_loc].pct_change().dropna()
            if len(ret_hist) >= lw_min_history:
                try:
                    lw = LedoitWolf().fit(ret_hist.values)
                    cov = lw.covariance_
                    ones = np.ones(n)
                    inv_cov = np.linalg.pinv(cov)
                    raw_w = inv_cov @ ones
                    raw_w = np.clip(raw_w, 0.05, None)
                    raw_w = np.clip(raw_w, None, 0.70 * raw_w.sum())
                    w_arr = raw_w / raw_w.sum()
                    weights = {code: float(w_arr[i]) for i, code in enumerate(target_codes)}
                except Exception:
                    weights = {c: 1.0 / n for c in target_codes}
            else:
                weights = {c: 1.0 / n for c in target_codes}
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


def calc_full_stats(nav: pd.Series, label: str = "") -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = cagr / downside if downside > 0 else 0
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    wins = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    losses = monthly[monthly < 0].abs().mean() if (monthly < 0).any() else 1
    pnl_ratio = wins / losses if losses > 0 else 0
    return {
        "标的": label,
        "年化收益": f"{cagr*100:.1f}%",
        "夏普": f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%",
        "Calmar": f"{calmar:.2f}",
        "Sortino": f"{sortino:.2f}",
        "月胜率": f"{win_rate:.1%}",
        "盈亏比": f"{pnl_ratio:.2f}",
        "_sharpe": sharpe,
        "_maxdd": max_dd,
        "_cagr": cagr,
    }


# ── 加载数据 ──────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
scores = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
close_is = close[close.index < split_date]
close_oos = close[close.index >= split_date]
rebal_is = [d for d in rebal_dates if d < split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
sc_is = scores[scores.index < split_date]
sc_oos = scores[scores.index >= split_date]

bench_nav = close[BENCHMARK].dropna()
bench_nav = bench_nav / bench_nav.iloc[0] * INIT_CASH

# ── 基线 ──────────────────────────────────────────────────

print("\n运行基线（等权）...")
nav_base = run_backtest(close, scores, rebal_dates, use_ledoit_wolf=False)
base_stats = calc_full_stats(nav_base, "基线（等权）")
base_sharpe = base_stats["_sharpe"]
print(f"  基线夏普={base_sharpe:.3f}  年化={base_stats['年化收益']}  回撤={base_stats['最大回撤']}")

# ── 1. 参数网格扫描 ───────────────────────────────────────

print("\n" + "=" * 80)
print(f"参数网格扫描：lw_min_history×{MIN_HISTORY_GRID} × hist_window×{HIST_WINDOW_GRID}")
print("=" * 80)

grid_results = []
navs_grid = {}
for mh in MIN_HISTORY_GRID:
    for hw in HIST_WINDOW_GRID:
        if mh > hw:
            continue  # min_history 不应超过历史窗口长度
        label = f"mh={mh},hw={hw}"
        nav = run_backtest(close, scores, rebal_dates, use_ledoit_wolf=True,
                            lw_min_history=mh, hist_window=hw)
        s = calc_full_stats(nav, label)
        grid_results.append({"lw_min_history": mh, "hist_window": hw, **s})
        navs_grid[label] = nav
        print(f"  {label:<18} 夏普={s['_sharpe']:.3f}  年化={s['年化收益']}  回撤={s['最大回撤']}")

n_trials = len(grid_results)
n_better = sum(1 for r in grid_results if r["_sharpe"] > base_sharpe)
best = max(grid_results, key=lambda r: r["_sharpe"])
print(f"\n测试组合数：{n_trials}")
print(f"超过基线({base_sharpe:.3f})的比例：{n_better}/{n_trials} = {n_better/n_trials:.0%}")
print(f"最优配置：lw_min_history={best['lw_min_history']}, hist_window={best['hist_window']}"
      f"  夏普={best['_sharpe']:.3f}（提升{best['_sharpe']-base_sharpe:+.3f}）")

best_label = f"mh={best['lw_min_history']},hw={best['hist_window']}"
nav_best = navs_grid[best_label]

# ── 全量指标对比（基线 vs 最优）────────────────────────────

print("\n" + "=" * 80)
print("全量指标对比：基线 vs 最优LW配置")
print("=" * 80)
df_cmp = pd.DataFrame([base_stats, calc_full_stats(nav_best, best_label)])
df_cmp = df_cmp[["标的", "年化收益", "夏普", "最大回撤", "Calmar", "Sortino", "月胜率", "盈亏比"]].set_index("标的")
print(df_cmp.to_string())

# ── 2. 逐年净值对比 ───────────────────────────────────────

print("\n" + "=" * 80)
print(f"逐年收益对比：基线 vs 最优LW配置（{best_label}）")
print("=" * 80)
all_years = sorted(set(nav_base.index.year))
print(f"{'年份':<6}  {'基线':>10}  {'最优LW':>10}  {'差值':>8}")
for yr in all_years:
    s = pd.Timestamp(f"{yr}-01-01"); e = pd.Timestamp(f"{yr}-12-31")
    seg_b = nav_base[(nav_base.index >= s) & (nav_base.index <= e)]
    seg_l = nav_best[(nav_best.index >= s) & (nav_best.index <= e)]
    if seg_b.empty or seg_l.empty:
        continue
    ret_b = seg_b.iloc[-1] / seg_b.iloc[0] - 1
    ret_l = seg_l.iloc[-1] / seg_l.iloc[0] - 1
    diff = ret_l - ret_b
    marker = " ←" if abs(diff) > 0.10 else ""
    print(f"{yr:<6}  {ret_b*100:>9.1f}%  {ret_l*100:>9.1f}%  {diff*100:>+7.1f}%{marker}")

# ── 3. 滚动3年窗口夏普稳健性 ─────────────────────────────

print("\n" + "=" * 80)
print("滚动3年夏普（基线 vs 最优LW配置）")
print("=" * 80)


def roll_sharpe(s):
    r = s.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0


window_days = 252 * 3
rolling_base = []
rolling_best = []
for i in range(window_days, len(nav_base)):
    seg_b = nav_base.iloc[i - window_days: i]
    seg_l = nav_best.iloc[i - window_days: i]
    rolling_base.append((nav_base.index[i], roll_sharpe(seg_b)))
    rolling_best.append((nav_base.index[i], roll_sharpe(seg_l)))

rs_base = pd.Series(dict(rolling_base))
rs_best = pd.Series(dict(rolling_best))
improvement = rs_best - rs_base
print(f"滚动3年夏普均值：基线={rs_base.mean():.2f}，最优LW={rs_best.mean():.2f}")
print(f"差值：均值={improvement.mean():+.3f}，std={improvement.std():.3f}，"
      f"最小={improvement.min():+.3f}，最大={improvement.max():+.3f}")
neg_periods = (improvement < 0).mean()
print(f"最优LW劣于基线的滚动窗口占比：{neg_periods:.1%}")

# ── 4. 最优参数 IS/OOS 验证 ───────────────────────────────

print("\n" + "=" * 80)
print(f"最优配置 IS/OOS 验证（{best_label}）")
print("=" * 80)
n_is = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=True,
                     lw_min_history=best["lw_min_history"], hist_window=best["hist_window"])
n_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_ledoit_wolf=True,
                      lw_min_history=best["lw_min_history"], hist_window=best["hist_window"])
n_is_base = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=False)
n_oos_base = run_backtest(close_oos, sc_oos, rebal_oos, use_ledoit_wolf=False)

si = calc_full_stats(n_is); so = calc_full_stats(n_oos)
si_b = calc_full_stats(n_is_base); so_b = calc_full_stats(n_oos_base)
decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
status = "通过" if decay >= 0.5 else "警告:过拟合"
print(f"基线   IS夏普={si_b['_sharpe']:.3f}  OOS夏普={so_b['_sharpe']:.3f}")
print(f"最优LW IS夏普={si['_sharpe']:.3f}  OOS夏普={so['_sharpe']:.3f}  OOS/IS={decay:.2f}  [{status}]")

print("\n完成。")
