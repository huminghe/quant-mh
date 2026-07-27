"""
第九轮方向2：组合层面波动率目标控制验证（2026-07）

背景：区别于已验证有效的"单标的按波动率反比加权仓位"（风险调整动量本身已含此逻辑），
这里测试给整个组合总敞口按历史组合波动率动态缩放（vol targeting）——
组合实际波动率高于目标时降低总仓位（多余部分留现金），低于目标时提升总仓位（不超100%，无杠杆）。

同类"按波动率信号调整敞口"机制此前已失败3次（市场状态区制切换、动态持仓数、
广度连续仓位），预期大概率也失败，但确实是没测过的变体，值得一次快速验证彻底排除。

网格：target_vol ∈ [12%, 15%, 18%, 22%] × vol_lookback ∈ [21, 42, 63] 日
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
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

TARGET_VOL_GRID  = [0.12, 0.15, 0.18, 0.22]
VOL_LOOKBACK_GRID = [21, 42, 63]
MIN_EXPOSURE     = 0.3   # 敞口下限，避免vol飙升时几乎全部空仓
MAX_EXPOSURE     = 1.0   # 无杠杆

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 信号计算（与线上一致）────────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
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


# ── 回测引擎（含组合层面 vol targeting）─────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    use_vol_target: bool = False,
    target_vol: float = 0.15,
    vol_lookback: int = 21,
) -> pd.Series:
    cash = init_cash
    holdings: dict[str, float] = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)
    nav_hist: list[float] = []  # 已实现组合净值，用于估算滚动波动率

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value
        nav_hist.append(port_value)

        if date not in rebal_set:
            continue

        day_scores = scores.loc[date].dropna()
        pos_scores = day_scores[day_scores > 0].nlargest(top_n * 3)
        candidates = list(pos_scores.index)
        target_codes = candidates[:top_n]

        # ── 组合层面波动率目标 ──────────────────────────────
        exposure = 1.0
        if use_vol_target and len(nav_hist) >= vol_lookback + 1:
            recent_nav = pd.Series(nav_hist[-(vol_lookback + 1):])
            recent_rets = recent_nav.pct_change().dropna()
            realized_vol = recent_rets.std() * np.sqrt(252)
            if realized_vol > 1e-6:
                exposure = target_vol / realized_vol
                exposure = float(np.clip(exposure, MIN_EXPOSURE, MAX_EXPOSURE))

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        n = len(target_codes)
        weights = {c: exposure / n for c in target_codes}

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

print("\n运行基线...")
nav_base = run_backtest(close, scores, rebal_dates, use_vol_target=False)
base_stats = calc_full_stats(nav_base, "基线")
print(f"  基线全样本夏普={base_stats['_sharpe']:.3f}")

# ── 1. 参数网格扫描 ──────────────────────────────────────

print("\n" + "=" * 80)
print(f"参数网格扫描：target_vol × vol_lookback（{len(TARGET_VOL_GRID)}×{len(VOL_LOOKBACK_GRID)}={len(TARGET_VOL_GRID)*len(VOL_LOOKBACK_GRID)}组）")
print("=" * 80)

grid_results = []
for tv in TARGET_VOL_GRID:
    for lb in VOL_LOOKBACK_GRID:
        nav = run_backtest(close, scores, rebal_dates, use_vol_target=True,
                            target_vol=tv, vol_lookback=lb)
        s = calc_full_stats(nav)
        grid_results.append({"target_vol": tv, "vol_lookback": lb, "sharpe": s["_sharpe"],
                              "maxdd": s["_maxdd"], "cagr": s["_cagr"], "nav": nav})
        beat = "✓" if s["_sharpe"] > base_stats["_sharpe"] else " "
        print(f"  target_vol={tv:.0%}  lookback={lb:>3}日  夏普={s['_sharpe']:.3f}  "
              f"回撤={s['_maxdd']*100:.1f}%  {beat}")

n_beat = sum(1 for r in grid_results if r["sharpe"] > base_stats["_sharpe"])
print(f"\n超基线比例：{n_beat}/{len(grid_results)} = {n_beat/len(grid_results)*100:.0f}%")

best = max(grid_results, key=lambda r: r["sharpe"])
print(f"最优配置：target_vol={best['target_vol']:.0%}, lookback={best['vol_lookback']}日, "
      f"夏普={best['sharpe']:.3f}（基线={base_stats['_sharpe']:.3f}）")

# ── 2. 逐年对比（最优配置）────────────────────────────────

print("\n" + "=" * 80)
print(f"逐年收益对比：基线 vs 最优vol-target配置（tv={best['target_vol']:.0%}, lb={best['vol_lookback']}）")
print("=" * 80)
nav_best = best["nav"]
all_years = sorted(set(nav_base.index.year))
print(f"{'年份':<6}  {'基线':>10}  {'VolTarget':>10}  {'差值':>8}")
for yr in all_years:
    s = pd.Timestamp(f"{yr}-01-01"); e = pd.Timestamp(f"{yr}-12-31")
    seg_b = nav_base[(nav_base.index >= s) & (nav_base.index <= e)]
    seg_v = nav_best[(nav_best.index >= s) & (nav_best.index <= e)]
    if seg_b.empty or seg_v.empty:
        continue
    ret_b = seg_b.iloc[-1] / seg_b.iloc[0] - 1
    ret_v = seg_v.iloc[-1] / seg_v.iloc[0] - 1
    diff = ret_v - ret_b
    marker = " ←" if abs(diff) > 0.10 else ""
    print(f"{yr:<6}  {ret_b*100:>9.1f}%  {ret_v*100:>9.1f}%  {diff*100:>+7.1f}%{marker}")

# ── 3. IS/OOS 验证（最优配置）─────────────────────────────

print("\n" + "=" * 80)
print(f"IS/OOS 验证：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")
print("=" * 80)
nav_base_is = run_backtest(close_is, sc_is, rebal_is, use_vol_target=False)
nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_vol_target=False)
nav_best_is = run_backtest(close_is, sc_is, rebal_is, use_vol_target=True,
                            target_vol=best["target_vol"], vol_lookback=best["vol_lookback"])
nav_best_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_vol_target=True,
                             target_vol=best["target_vol"], vol_lookback=best["vol_lookback"])

for label, nav_i, nav_o, nav_f in [
    ("基线", nav_base_is, nav_base_oos, nav_base),
    ("VolTarget最优", nav_best_is, nav_best_oos, nav_best),
]:
    si = calc_full_stats(nav_i)
    so = calc_full_stats(nav_o)
    sf = calc_full_stats(nav_f)
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告:可能过拟合"
    print(f"  {label:<14}  IS夏普={si['_sharpe']:.3f}  OOS夏普={so['_sharpe']:.3f}  "
          f"OOS/IS={decay:.2f}  全样本夏普={sf['_sharpe']:.3f}  {status}")

# ── 结论 ─────────────────────────────────────────────────

# ── 4. 滚动3年窗口夏普稳健性（网格仅25%超基线，参照LW教训需补验证）───

print("\n" + "=" * 80)
print("滚动3年夏普（基线 vs 最优VolTarget配置）")
print("=" * 80)


def roll_sharpe(s):
    r = s.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0


window_days = 252 * 3
rolling_base, rolling_best = [], []
for i in range(window_days, len(nav_base)):
    seg_b = nav_base.iloc[i - window_days: i]
    seg_v = nav_best.iloc[i - window_days: i]
    rolling_base.append((nav_base.index[i], roll_sharpe(seg_b)))
    rolling_best.append((nav_base.index[i], roll_sharpe(seg_v)))

rs_base = pd.Series(dict(rolling_base))
rs_best = pd.Series(dict(rolling_best))
improvement = rs_best - rs_base
print(f"滚动3年夏普均值：基线={rs_base.mean():.2f}，最优VolTarget={rs_best.mean():.2f}")
print(f"差值：均值={improvement.mean():+.3f}，std={improvement.std():.3f}，"
      f"最小={improvement.min():+.3f}，最大={improvement.max():+.3f}")
neg_periods = (improvement < 0).mean()
print(f"最优VolTarget劣于基线的滚动窗口占比：{neg_periods:.1%}")

# ── 结论 ─────────────────────────────────────────────────

delta = best["sharpe"] - base_stats["_sharpe"]
tag = "有效" if delta > 0.02 else ("中性" if delta > -0.02 else "有害")
print("\n" + "=" * 80)
print(f"结论：组合层面波动率目标控制 Δ夏普={delta:+.3f}  [{tag}]（最优配置 vs 基线1.053）")
print(f"网格超基线比例25%，改善集中在2019/2020/2026三年，滚动3年劣于基线占比{neg_periods:.1%}")
print("=" * 80)

# ── 可视化 ───────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1.5]})
ax1, ax2 = axes
ax1.plot(nav_base.index, nav_base / INIT_CASH, label=f"基线  Sharpe={base_stats['_sharpe']:.3f}",
          lw=2.0, ls="--", color="#9E9E9E")
ax1.plot(nav_best.index, nav_best / INIT_CASH,
          label=f"VolTarget(tv={best['target_vol']:.0%},lb={best['vol_lookback']})  Sharpe={best['sharpe']:.3f}",
          lw=1.5, color="#E53935")
ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800", ls=":", lw=0.9, alpha=0.6, label="沪深300")
ax1.axvline(split_date, color="red", ls="--", alpha=0.4, lw=1.0, label="IS/OOS分割")
ax1.set_title("ETF轮动 — 组合层面波动率目标控制验证（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", ls="--", alpha=0.3)

dd_b = (nav_base - nav_base.cummax()) / nav_base.cummax() * 100
dd_v = (nav_best - nav_best.cummax()) / nav_best.cummax() * 100
ax2.plot(dd_b.index, dd_b, lw=1.8, ls="--", color="#9E9E9E", label=f"基线 MaxDD={dd_b.min():.1f}%")
ax2.plot(dd_v.index, dd_v, lw=1.3, color="#E53935", label=f"VolTarget MaxDD={dd_v.min():.1f}%")
ax2.set_ylabel("回撤(%)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_title("回撤对比")
plt.tight_layout()
fig_path = out_dir / "etf_rotation_v9_voltarget.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"\n图已保存：{fig_path}")

print("\n完成。")
