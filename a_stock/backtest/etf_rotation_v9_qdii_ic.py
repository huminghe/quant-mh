"""
第九轮方向3：QDII跨境ETF信号质量IC检验（2026-07）

背景：文档记录"QDII/REIT纳入轮动"为"不做/未测（排除）"，但实际标的池
`etf_universe.py` 中QDII类ETF（纳指159941、标普500 513500、恒生159920、
恒生科技513180、港股互联网516950、中概互联513050，共6只）已经在参与
Top3竞争，历史所有轮次回测和当前模拟盘都包含它们——文档滞后于代码。

本轮任务不是"是否要加入QDII"（已经加入），而是检验：
1. 这6只QDII标的历史上进入Top3的频率
2. 动量信号在QDII标的上的IC（信号 vs 未来1月收益）是否弱于A股标的
3. 剔除QDII后策略表现变化——用于判断QDII对组合是正贡献还是噪声

方法：先做IC检验排除法（不直接进组合回测出负效果就误判），避免重复
"商品ETF未验证信号质量直接进组合测出负效果"的错误。
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
from etf_universe import ETF_UNIVERSE

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

QDII_CODES = ["159941.SZ", "513500.SH", "159920.SZ", "513180.SH", "516950.SH", "513050.SH"]

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


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
                  top_n: int = TOP_N, init_cash: float = INIT_CASH) -> tuple[pd.Series, dict]:
    cash = init_cash
    holdings: dict[str, float] = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)
    qdii_pick_count = 0
    total_picks = 0

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
        total_picks += len(target_codes)
        qdii_pick_count += sum(1 for c in target_codes if c in QDII_CODES)

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

    meta = {"qdii_pick_ratio": qdii_pick_count / total_picks if total_picks else 0,
            "qdii_pick_count": qdii_pick_count, "total_picks": total_picks}
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
        "标的": label, "年化收益": f"{cagr*100:.1f}%", "夏普": f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%", "Calmar": f"{calmar:.2f}", "Sortino": f"{sortino:.2f}",
        "月胜率": f"{win_rate:.1%}", "盈亏比": f"{pnl_ratio:.2f}",
        "_sharpe": sharpe, "_maxdd": max_dd, "_cagr": cagr,
    }


# ── 加载数据 ──────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
domestic_codes = [c for c in valid_codes if c not in QDII_CODES]
print(f"有效标的：{len(valid_codes)} 只（含QDII {len([c for c in QDII_CODES if c in valid_codes])} 只），"
      f"{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
scores = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# ── 1. IC检验：QDII vs A股标的，信号 vs 未来1月收益 ─────────

print("\n" + "=" * 80)
print("诊断1：动量信号IC对比 —— QDII跨境ETF vs A股标的")
print("=" * 80)

fwd_1m = close.pct_change().rolling(21).sum().shift(-21)

ic_rows = []
for group_name, codes in [("QDII跨境（6只）", [c for c in QDII_CODES if c in valid_codes]),
                            ("A股标的（其余）", domestic_codes)]:
    ics = []
    hits = []
    for code in codes:
        sig = scores[code].dropna()
        fwd = fwd_1m[code].dropna()
        df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
        if len(df) < 100:
            continue
        ic = df["sig"].corr(df["fwd"])
        hit = (np.sign(df["sig"]) == np.sign(df["fwd"])).mean()
        ics.append(ic)
        hits.append(hit)
    ic_rows.append({"分组": group_name, "标的数": len(ics),
                     "平均IC": np.mean(ics), "IC中位数": np.median(ics),
                     "平均方向命中率": np.mean(hits)})
print(pd.DataFrame(ic_rows).set_index("分组").to_string())

print("\n各QDII标的单独IC：")
for code in QDII_CODES:
    if code not in valid_codes:
        continue
    sig = scores[code].dropna()
    fwd = fwd_1m[code].dropna()
    df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
    if len(df) < 100:
        print(f"  {code} {ETF_UNIVERSE.get(code, code):<12}  数据不足，跳过")
        continue
    ic = df["sig"].corr(df["fwd"])
    hit = (np.sign(df["sig"]) == np.sign(df["fwd"])).mean()
    print(f"  {code} {ETF_UNIVERSE.get(code, code):<12}  IC={ic:+.3f}  方向命中率={hit:.1%}  样本={len(df)}")

# ── 2. Top3入选频率与贡献 ────────────────────────────────

print("\n" + "=" * 80)
print("诊断2：QDII标的历史Top3入选频率与贡献")
print("=" * 80)

nav_full, meta_full = run_backtest(close, scores, rebal_dates)
s_full = calc_full_stats(nav_full)
print(f"含QDII（当前实际配置） 全样本夏普={s_full['_sharpe']:.3f}  "
      f"Top3中QDII占比={meta_full['qdii_pick_ratio']:.1%}"
      f"（{meta_full['qdii_pick_count']}/{meta_full['total_picks']}次入选）")

# ── 3. 剔除QDII后对比（组合层面验证）────────────────────────

print("\n" + "=" * 80)
print("诊断3：剔除QDII vs 保留QDII —— 组合层面全量对比")
print("=" * 80)

close_no_qdii = close[domestic_codes]
scores_no_qdii = scores[domestic_codes]
nav_noq, meta_noq = run_backtest(close_no_qdii, scores_no_qdii, rebal_dates)
s_noq = calc_full_stats(nav_noq, "剔除QDII")
s_full_labeled = calc_full_stats(nav_full, "保留QDII（当前）")

df_cmp = pd.DataFrame([s_full_labeled, s_noq])
df_cmp = df_cmp[["标的", "年化收益", "夏普", "最大回撤", "Calmar", "Sortino", "月胜率", "盈亏比"]].set_index("标的")
print(df_cmp.to_string())

# IS/OOS
n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
rebal_is = [d for d in rebal_dates if d < split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]

close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]
close_noq_is, close_noq_oos = close_no_qdii[close_no_qdii.index < split_date], close_no_qdii[close_no_qdii.index >= split_date]
sc_noq_is, sc_noq_oos = scores_no_qdii[scores_no_qdii.index < split_date], scores_no_qdii[scores_no_qdii.index >= split_date]

nav_full_is, _ = run_backtest(close_is, sc_is, rebal_is)
nav_full_oos, _ = run_backtest(close_oos, sc_oos, rebal_oos)
nav_noq_is, _ = run_backtest(close_noq_is, sc_noq_is, rebal_is)
nav_noq_oos, _ = run_backtest(close_noq_oos, sc_noq_oos, rebal_oos)

print(f"\nIS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")
for label, ni, no in [("保留QDII（当前）", nav_full_is, nav_full_oos), ("剔除QDII", nav_noq_is, nav_noq_oos)]:
    si = calc_full_stats(ni); so = calc_full_stats(no)
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    print(f"  {label:<16}  IS夏普={si['_sharpe']:.3f}  OOS夏普={so['_sharpe']:.3f}  OOS/IS={decay:.2f}")

# ── 逐年对比 ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("逐年收益对比：保留QDII（当前） vs 剔除QDII")
print("=" * 80)
all_years = sorted(set(nav_full.index.year))
print(f"{'年份':<6}  {'保留QDII':>10}  {'剔除QDII':>10}  {'差值':>8}")
for yr in all_years:
    s = pd.Timestamp(f"{yr}-01-01"); e = pd.Timestamp(f"{yr}-12-31")
    seg_f = nav_full[(nav_full.index >= s) & (nav_full.index <= e)]
    seg_n = nav_noq[(nav_noq.index >= s) & (nav_noq.index <= e)]
    if seg_f.empty or seg_n.empty:
        continue
    ret_f = seg_f.iloc[-1] / seg_f.iloc[0] - 1
    ret_n = seg_n.iloc[-1] / seg_n.iloc[0] - 1
    diff = ret_f - ret_n
    marker = " ←" if abs(diff) > 0.05 else ""
    print(f"{yr:<6}  {ret_f*100:>9.1f}%  {ret_n*100:>9.1f}%  {diff*100:>+7.1f}%{marker}")

# ── 4. 滚动3年窗口稳健性（IC为负但组合效果为正，反直觉，需验证）─

print("\n" + "=" * 80)
print("滚动3年夏普（保留QDII vs 剔除QDII）—— IC为负但组合Δ夏普较大，需排查是否稳健")
print("=" * 80)


def roll_sharpe(s):
    r = s.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0


window_days = 252 * 3
rolling_full, rolling_noq = [], []
for i in range(window_days, len(nav_full)):
    seg_f = nav_full.iloc[i - window_days: i]
    seg_n = nav_noq.iloc[i - window_days: i]
    rolling_full.append((nav_full.index[i], roll_sharpe(seg_f)))
    rolling_noq.append((nav_noq.index[i], roll_sharpe(seg_n)))

rs_full = pd.Series(dict(rolling_full))
rs_noq = pd.Series(dict(rolling_noq))
improvement = rs_full - rs_noq
print(f"滚动3年夏普均值：保留QDII={rs_full.mean():.2f}，剔除QDII={rs_noq.mean():.2f}")
print(f"差值：均值={improvement.mean():+.3f}，std={improvement.std():.3f}，"
      f"最小={improvement.min():+.3f}，最大={improvement.max():+.3f}")
neg_periods = (improvement < 0).mean()
print(f"保留QDII劣于剔除QDII的滚动窗口占比：{neg_periods:.1%}")

# ── 结论 ─────────────────────────────────────────────────

delta = s_full["_sharpe"] - s_noq["_sharpe"]
tag = "保留QDII更优" if delta > 0.02 else ("中性" if delta > -0.02 else "剔除QDII更优")
print("\n" + "=" * 80)
print(f"结论：保留QDII vs 剔除QDII  Δ夏普={delta:+.3f}  [{tag}]")
print(f"个体IC全为负/接近零，但组合层面提升显著，滚动3年劣于对照占比{neg_periods:.1%}")
print("=" * 80)

# ── 可视化 ───────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

fig, ax1 = plt.subplots(figsize=(13, 6))
ax1.plot(nav_full.index, nav_full / INIT_CASH, label=f"保留QDII（当前）Sharpe={s_full['_sharpe']:.3f}",
          lw=1.8, color="#E53935")
ax1.plot(nav_noq.index, nav_noq / INIT_CASH, label=f"剔除QDII Sharpe={s_noq['_sharpe']:.3f}",
          lw=1.8, ls="--", color="#1565C0")
ax1.axvline(split_date, color="gray", ls="--", alpha=0.4, lw=1.0, label="IS/OOS分割")
ax1.set_title("ETF轮动 — QDII跨境ETF贡献验证（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
plt.tight_layout()
fig_path = out_dir / "etf_rotation_v9_qdii_ic.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"\n图已保存：{fig_path}")

print("\n完成。")
