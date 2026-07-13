"""
第九轮方向1：季度调仓验证（2026-07）

背景：当前月度调仓年化换手约1300%，年化成本约22%（详见 etf_rotation_v5_new_directions.py
方向5换手统计）。双周调仓已验证有害（夏普0.23）。季度调仓是完全没测过的反方向，
换手可降至约1/3，只要信号衰减不太快就有正收益空间。

对比：月度（基线）vs 双月 vs 季度调仓，全量指标 + 换手/成本 + 逐年对比 + IS/OOS。
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
ROUND_TRIP_COST  = 0.00164  # 佣金+滑点，单次完整回合

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


def get_rebalance_dates(index: pd.DatetimeIndex, freq_months: int = 1) -> list:
    """freq_months=1 月度，2 双月，3 季度：每N个月取第一个交易日"""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    month_starts = df.groupby("ym").apply(lambda x: x.index[0]).sort_index()
    if freq_months == 1:
        return month_starts.tolist()
    return month_starts.iloc[::freq_months].tolist()


# ── 回测引擎 ─────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    track_turnover: bool = False,
) -> tuple[pd.Series, dict]:
    cash = init_cash
    holdings: dict[str, float] = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    turnover_events: list[float] = []

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

        trade_value = 0.0

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    sell_val = holdings[code] * price
                    cash += sell_val * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    if track_turnover:
                        trade_value += sell_val
                del holdings[code]

        if not target_codes:
            if track_turnover and port_value > 0:
                turnover_events.append(trade_value / port_value)
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
                        if track_turnover:
                            trade_value += buy_shares * buy_price
            elif diff < -price * 100:
                sell_shares = int(-diff / price / 100) * 100
                if sell_shares > 0 and current_shares >= sell_shares:
                    cash += sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares
                    if track_turnover:
                        trade_value += sell_shares * price

        if track_turnover and port_value > 0:
            turnover_events.append(trade_value / port_value)

    meta = {}
    if track_turnover and turnover_events:
        n_rebal = len(turnover_events)
        avg_to = np.mean(turnover_events)
        years = (close.index[-1] - close.index[0]).days / 365.25
        meta["n_rebal"] = n_rebal
        meta["avg_turnover_per_rebal"] = avg_to
        meta["annual_turnover"] = avg_to * n_rebal / years
        meta["annual_cost_est"] = meta["annual_turnover"] * ROUND_TRIP_COST

    return nav_series.dropna(), meta


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


def yearly_return(nav: pd.Series, year: int) -> float | None:
    s = pd.Timestamp(f"{year}-01-01"); e = pd.Timestamp(f"{year}-12-31")
    seg = nav[(nav.index >= s) & (nav.index <= e)]
    if seg.empty:
        return None
    return seg.iloc[-1] / seg.iloc[0] - 1


# ── 加载数据 ──────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
scores = calc_all_scores(close)

n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
close_is = close[close.index < split_date]
close_oos = close[close.index >= split_date]
sc_is = scores[scores.index < split_date]
sc_oos = scores[scores.index >= split_date]

bench_nav = close[BENCHMARK].dropna()
bench_nav = bench_nav / bench_nav.iloc[0] * INIT_CASH

# ── 运行三种调仓频率 ─────────────────────────────────────

FREQS = [
    ("月度（基线）", 1),
    ("双月", 2),
    ("季度", 3),
]

print("\n运行回测...")
navs_full, navs_is, navs_oos, metas = {}, {}, {}, {}
for label, freq in FREQS:
    rebal_all = get_rebalance_dates(close.index, freq)
    rebal_is  = [d for d in rebal_all if d < split_date]
    rebal_oos = [d for d in rebal_all if d >= split_date]

    nav_f, meta = run_backtest(close, scores, rebal_all, track_turnover=True)
    nav_i, _    = run_backtest(close_is, sc_is, rebal_is)
    nav_o, _    = run_backtest(close_oos, sc_oos, rebal_oos)

    navs_full[label] = nav_f
    navs_is[label]   = nav_i
    navs_oos[label]  = nav_o
    metas[label]     = meta
    s = calc_full_stats(nav_f)
    print(f"  {label:<12} 全样本夏普={s['_sharpe']:.3f}  年化换手={meta.get('annual_turnover', 0)*100:.0f}%")

# ── 全量指标对比 ─────────────────────────────────────────

print("\n" + "=" * 80)
print("全样本全量指标对比（2016-2026）")
print("=" * 80)
rows_full = [calc_full_stats(navs_full[l], l) for l, _ in FREQS]
df_full = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_full]).set_index("标的")
print(df_full.to_string())

print("\n" + "=" * 80)
print("换手率与成本对比")
print("=" * 80)
for label, _ in FREQS:
    m = metas[label]
    if m:
        print(f"  {label:<12} 调仓次数={m['n_rebal']:<4} 年化换手={m['annual_turnover']*100:>7.1f}%  "
              f"年化成本估算={m['annual_cost_est']*100:>5.2f}%")

# ── 逐年对比 ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("逐年收益对比")
print("=" * 80)
all_years = sorted(set(close.index.year))
header = f"{'年份':<6}" + "".join(f"{label:>12}" for label, _ in FREQS)
print(header)
for yr in all_years:
    row = f"{yr:<6}"
    for label, _ in FREQS:
        r = yearly_return(navs_full[label], yr)
        row += f"{r*100:>11.1f}%" if r is not None else f"{'--':>12}"
    print(row)

# ── IS/OOS 验证 ───────────────────────────────────────────

print("\n" + "=" * 80)
print(f"IS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")
print("=" * 80)
is_oos_rows = []
for label, _ in FREQS:
    si = calc_full_stats(navs_is[label])
    so = calc_full_stats(navs_oos[label])
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告:可能过拟合"
    is_oos_rows.append({
        "配置": label,
        "IS夏普": f"{si['_sharpe']:.3f}",
        "OOS夏普": f"{so['_sharpe']:.3f}",
        "OOS/IS": f"{decay:.2f}",
        "全样本夏普": f"{calc_full_stats(navs_full[label])['_sharpe']:.3f}",
        "状态": status,
    })
print(pd.DataFrame(is_oos_rows).set_index("配置").to_string())

# ── 结论汇总 ─────────────────────────────────────────────

base_sharpe = calc_full_stats(navs_full["月度（基线）"])["_sharpe"]
print("\n" + "=" * 80)
print("结论汇总（vs 月度基线夏普 {:.3f}）".format(base_sharpe))
print("=" * 80)
for row in rows_full:
    label = row["标的"]
    if label == "月度（基线）":
        continue
    delta = row["_sharpe"] - base_sharpe
    tag = "有效" if delta > 0.02 else ("中性" if delta > -0.02 else "有害")
    print(f"  {label:<12}  Δ夏普={delta:+.3f}  [{tag}]")

# ── 可视化 ───────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

colors = {"月度（基线）": "#9E9E9E", "双月": "#1565C0", "季度": "#E53935"}

fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1.5]})
ax1, ax2 = axes
for label, _ in FREQS:
    nav = navs_full[label]
    s = calc_full_stats(nav)
    ax1.plot(nav.index, nav / INIT_CASH, label=f"{label}  Sharpe={s['_sharpe']:.3f}",
              lw=2.0 if label == "月度（基线）" else 1.5,
              ls="--" if label == "月度（基线）" else "-",
              color=colors[label])
ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800", ls=":", lw=0.9, alpha=0.6, label="沪深300")
ax1.axvline(split_date, color="red", ls="--", alpha=0.4, lw=1.0, label="IS/OOS分割")
ax1.set_title("ETF轮动 — 调仓频率对比（月度/双月/季度，2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", ls="--", alpha=0.3)

for label, _ in FREQS:
    nav = navs_full[label]
    dd = (nav - nav.cummax()) / nav.cummax() * 100
    ax2.plot(dd.index, dd, lw=1.8 if label == "月度（基线）" else 1.3,
              ls="--" if label == "月度（基线）" else "-",
              color=colors[label], label=f"{label} MaxDD={dd.min():.1f}%")
ax2.set_ylabel("回撤(%)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_title("回撤对比")
plt.tight_layout()
fig_path = out_dir / "etf_rotation_v9_quarterly.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"\n图已保存：{fig_path}")

print("\n完成。")
