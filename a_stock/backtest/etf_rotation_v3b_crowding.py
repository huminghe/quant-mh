"""
ETF 轮动方向B：行业拥挤度（价格相关性代理）测试（2026-07）

拥挤度定义：
  每只 ETF 与宇宙内其余 ETF 的滚动平均两两相关性（60日）。
  高相关 → 行业联动强 → 动量信号同质化 → 拥挤。

信号修正方案：
  B1: 得分 × (1 - 拥挤度分位数)    — 连续打折
  B2: 拥挤度 > 历史80%分位 时得分 × 0  — 硬过滤（完全剔除）
  B3: 拥挤度 > 历史80%分位 时得分 × 0.5 — 软过滤（半仓惩罚）

回测输出：
  - 全样本对比表（含基线）
  - IS/OOS 验证
  - 净值曲线图
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
CORR_WINDOW      = 60    # 拥挤度计算窗口（交易日）
CORR_HIST_WINDOW = 252   # 拥挤度历史分位数计算窗口（约1年）

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 信号计算 ───────────────────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(
    close_matrix: pd.DataFrame,
    window: int = MOMENTUM_WINDOW,
    risk_adj: bool = True,
    risk_vol_window: int = RISK_VOL_WINDOW,
) -> pd.DataFrame:
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            pw  = series.iloc[i - window: i]
            raw = momentum_score(pw)
            if risk_adj and i >= risk_vol_window:
                rets = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol  = rets.std() * np.sqrt(252)
                raw  = raw / vol if vol > 1e-6 else raw
            score_series.iloc[i] = raw
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


def calc_crowding(
    close: pd.DataFrame,
    corr_window: int = CORR_WINDOW,
    hist_window: int = CORR_HIST_WINDOW,
) -> pd.DataFrame:
    """
    计算每只 ETF 在每个交易日的拥挤度分位数（0~1）。
    拥挤度 = 该 ETF 与宇宙内其余 ETF 的平均两两相关系数（滚动60日）。
    分位数 = 过去 hist_window 日内该 ETF 拥挤度的历史分位（避免截面比较带来的行业偏差）。

    返回：与 close 同索引的 DataFrame，values 是拥挤度历史分位数（0~1）。
    """
    # 去掉基准，只用行业/宽基ETF
    codes = [c for c in close.columns if c != BENCHMARK]
    ret = close[codes].pct_change()

    print(f"  计算 {len(codes)} 只ETF × {len(close)} 日的滚动相关性，窗口={corr_window}日...")

    # 逐日计算每只ETF与其余ETF的平均相关系数
    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)

    for i in range(corr_window, len(close.index)):
        ret_win = ret.iloc[i - corr_window: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        # 对角线为1，排除自身
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(
            np.nanmean(corr_arr, axis=1),
            index=ret_win.columns,
        )
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values

    # 转为历史分位数（每只ETF时序分位，防止行业间系统性差异干扰）
    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(hist_window + corr_window, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - hist_window: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna()
            c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()

    return crowding_pct


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 回测主逻辑 ─────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    crowding_pct: pd.DataFrame = None,
    crowding_mode: str = "none",   # "none" / "continuous" / "hard" / "soft"
    crowding_threshold: float = 0.80,
    crowding_soft_factor: float = 0.5,
) -> pd.Series:
    """
    月度轮换回测，支持三种拥挤度修正方式：
    - none:       不修正（基线）
    - continuous: 得分 × (1 - 拥挤度分位数)
    - hard:       拥挤度分位数 > threshold → 得分归零（完全剔除）
    - soft:       拥挤度分位数 > threshold → 得分 × soft_factor
    """
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

        day_scores = scores.loc[date].dropna().copy()

        # 拥挤度修正
        if crowding_mode != "none" and crowding_pct is not None and date in crowding_pct.index:
            day_crowding = crowding_pct.loc[date]
            for code in day_scores.index:
                if code not in day_crowding.index or pd.isna(day_crowding[code]):
                    continue
                pct = day_crowding[code]
                if crowding_mode == "continuous":
                    day_scores[code] *= (1.0 - pct)
                elif crowding_mode == "hard":
                    if pct > crowding_threshold:
                        day_scores[code] = 0.0
                elif crowding_mode == "soft":
                    if pct > crowding_threshold:
                        day_scores[code] *= crowding_soft_factor

        pos_scores  = day_scores[day_scores > 0].nlargest(top_n)
        target_codes = list(pos_scores.index)

        # 卖出不在目标中的持仓
        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE/2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        # 等权分配
        n = len(target_codes)
        weights = {c: 1.0 / n for c in target_codes}

        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            buy_price      = price * (1 + SLIPPAGE/2)
            target_value   = port_value * weights[code]
            current_shares = holdings.get(code, 0)
            current_value  = current_shares * price
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
                    cash += sell_shares * price * (1 - SLIPPAGE/2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


def calc_stats(nav: pd.Series) -> dict:
    rets  = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr  = (nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd,
            "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0}


# ── 主流程 ─────────────────────────────────────────────────

print("加载价格数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

print("\n计算动量得分...")
scores = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

print("\n计算行业拥挤度（相关性代理）...")
crowding_pct = calc_crowding(close)

# 拥挤度统计
valid_crowding = crowding_pct.stack().dropna()
print(f"  拥挤度分位数统计：均值={valid_crowding.mean():.2f}，std={valid_crowding.std():.2f}")
print(f"  >80%分位的占比：{(valid_crowding > 0.8).mean():.1%}（预期约20%）")

# IS/OOS 分割
n_days    = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
print(f"\nIS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")

bench     = close[BENCHMARK].dropna()
bench_nav = bench / bench.iloc[0] * INIT_CASH

print("\n开始回测（全样本）...")

CONFIGS = [
    ("基线（无拥挤度修正）",         "none",       0.80, 0.5),
    ("B1: 连续打折（×(1-分位数)）",  "continuous", 0.80, 0.5),
    ("B2: 硬过滤（>80%分位剔除）",   "hard",       0.80, 0.5),
    ("B3: 软过滤（>80%分位×0.5）",  "soft",       0.80, 0.5),
    ("B2b: 硬过滤（>70%分位）",     "hard",       0.70, 0.5),
    ("B3b: 软过滤（>70%分位×0.3）", "soft",       0.70, 0.3),
]

rows = []
navs = {}
for label, mode, threshold, soft_f in CONFIGS:
    nav = run_backtest(close, scores, rebal_dates,
                       crowding_pct=crowding_pct,
                       crowding_mode=mode,
                       crowding_threshold=threshold,
                       crowding_soft_factor=soft_f)
    s = calc_stats(nav)
    rows.append({
        "配置":     label,
        "年化收益": f"{s['CAGR']*100:.1f}%",
        "夏普":     f"{s['Sharpe']:.2f}",
        "最大回撤": f"{s['MaxDD']*100:.1f}%",
        "Calmar":   f"{s['Calmar']:.2f}",
    })
    navs[label] = nav
    print(f"  {label:<36}  夏普={s['Sharpe']:.2f}  年化={s['CAGR']*100:.1f}%  回撤={s['MaxDD']*100:.1f}%")

result_df = pd.DataFrame(rows).set_index("配置")
print("\n" + "=" * 75)
print("全样本对比（2016-2026）")
print("=" * 75)
print(result_df.to_string())

# ── IS/OOS 验证 ────────────────────────────────────────────

close_is  = close[close.index <  split_date]
close_oos = close[close.index >= split_date]
rebal_is  = [d for d in rebal_dates if d <  split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
sc_is     = scores[scores.index <  split_date]
sc_oos    = scores[scores.index >= split_date]
cp_is     = crowding_pct[crowding_pct.index <  split_date]
cp_oos    = crowding_pct[crowding_pct.index >= split_date]

print(f"\n\nIS/OOS 验证")
print("=" * 75)

is_oos_rows = []
for label, mode, threshold, soft_f in CONFIGS:
    ni  = run_backtest(close_is,  sc_is,  rebal_is,  crowding_pct=cp_is,
                       crowding_mode=mode, crowding_threshold=threshold, crowding_soft_factor=soft_f)
    no  = run_backtest(close_oos, sc_oos, rebal_oos, crowding_pct=cp_oos,
                       crowding_mode=mode, crowding_threshold=threshold, crowding_soft_factor=soft_f)
    si  = calc_stats(ni)
    so  = calc_stats(no)
    decay = so["Sharpe"] / si["Sharpe"] if si["Sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告:过拟合"
    is_oos_rows.append({
        "配置": label,
        "IS夏普":  f"{si['Sharpe']:.2f}",
        "IS年化":  f"{si['CAGR']*100:.1f}%",
        "IS回撤":  f"{si['MaxDD']*100:.1f}%",
        "OOS夏普": f"{so['Sharpe']:.2f}",
        "OOS年化": f"{so['CAGR']*100:.1f}%",
        "OOS回撤": f"{so['MaxDD']*100:.1f}%",
        "OOS/IS":  f"{decay:.2f}",
        "状态": status,
    })
    print(f"\n{label}")
    print(f"  IS ：夏普={si['Sharpe']:.2f}  年化={si['CAGR']*100:.1f}%  回撤={si['MaxDD']*100:.1f}%")
    print(f"  OOS：夏普={so['Sharpe']:.2f}  年化={so['CAGR']*100:.1f}%  回撤={so['MaxDD']*100:.1f}%")
    print(f"  OOS/IS={decay:.2f}  [{status}]")

print("\n\n汇总表：")
print(pd.DataFrame(is_oos_rows).set_index("配置").to_string())

# ── 净值曲线图 ────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1]})
ax1, ax2 = axes

colors = ["#9E9E9E", "#2196F3", "#E53935", "#43A047", "#7B1FA2", "#F57F17"]
for (label, *_), color in zip(CONFIGS, colors):
    nav = navs[label]
    lw  = 2.2 if "基线" in label else 1.4
    ax1.plot(nav.index, nav / INIT_CASH, label=label, color=color, linewidth=lw)

ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800",
         linestyle="--", lw=1.2, alpha=0.7, label="沪深300买持")
ax1.axvline(split_date, color="red", linestyle="--", alpha=0.5, lw=1)
ax1.set_title("ETF轮动 方向B（行业拥挤度修正）净值对比（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

# 回撤对比
base_nav = navs["基线（无拥挤度修正）"]
base_dd  = (base_nav - base_nav.cummax()) / base_nav.cummax() * 100
ax2.fill_between(base_dd.index, base_dd, 0, alpha=0.35, color="#9E9E9E", label=f"基线 MaxDD={base_dd.min():.1f}%")

best_label = max(
    [r["配置"] for r in rows if r["配置"] != "基线（无拥挤度修正）"],
    key=lambda l: float(next(r["夏普"] for r in rows if r["配置"] == l)),
)
best_nav = navs[best_label]
best_dd  = (best_nav - best_nav.cummax()) / best_nav.cummax() * 100
ax2.fill_between(best_dd.index, best_dd, 0, alpha=0.35, color="#2196F3",
                 label=f"最优({best_label[:10]}...) MaxDD={best_dd.min():.1f}%")
ax2.set_ylabel("回撤(%)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "etf_rotation_v3_crowding.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n净值曲线图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
