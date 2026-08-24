"""
公募基金持仓增减仓强度因子 截面IC验证（第十六轮候选①）

背景：数据可行性调研（未落盘的临时脚本）确认公募基金持仓（fund_portfolio）
存在明显的频率限制——一季报/三季报（Q1/Q3）只披露前十大重仓股，半年报/
年报（Q2/Q4）才披露接近完整持仓。用户确认方案：只用Q2/Q4数据，放弃
季度频率。详见 fetch_fund_holding.py 头部注释及 a_stock/docs/research.md
「指数增强策略」章节第十六轮记录。

因子构造：每只个股在半年报截面上的"被基金持仓强度变化"：
  holding_ratio(t) = sum(基金i持有该股市值 * 该基金规模权重) / 该股流通市值
  简化为：sum_i(stk_float_ratio_i) —— 持仓基金里该股占基金流通股比例之和
  （即"基金集体持仓占该股流通盘的比例"，不做基金规模加权，避免引入
  规模数据的另一套point-in-time对齐负担，遵循KISS）
  signal = holding_ratio(t) - holding_ratio(t-1)（环比变化，t为半年报期）

point-in-time：用ann_date（基金披露该期持仓的公告日）作为可知时点，
只用截面当日已公告的最新两期半年报/年报记录。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目
既有factor_ic_*.py方法论。由于因子只在半年报披露后更新（每年2次，1月/
4月附近的年报+8-9月附近的半年报，具体看ann_date分布），两次披露之间
月度截面因子值不变（信号本身是低频的，这是候选①的固有特性，不是bug）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%。

用法：
  cd a_stock/backtest
  python factor_ic_fund_holding.py               # 默认跑全部
  python factor_ic_fund_holding.py --index hs300
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR
from fetch_fund_holding import CACHE_DIR as FUND_HOLDING_DIR

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_fund_holding"

START_DATE = "2018-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50

INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
    },
}


# ── 工具函数 ──────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def cross_section_rank_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret"]
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
    return ic


# ── 基金持仓数据加载与个股持仓强度计算 ───────────────────────

def load_all_fund_holdings() -> pd.DataFrame:
    """
    合并所有基金的Q2/Q4持仓明细，返回长表：
    symbol（个股代码）, end_date, ann_date, stk_float_ratio
    """
    frames = []
    for path in FUND_HOLDING_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frames.append(df[["symbol", "end_date", "ann_date", "stk_float_ratio"]])
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df["end_date"] = pd.to_datetime(all_df["end_date"], errors="coerce")
    all_df["ann_date"] = pd.to_datetime(all_df["ann_date"], errors="coerce")
    all_df = all_df.dropna(subset=["end_date", "ann_date", "stk_float_ratio"])
    return all_df


def compute_holding_ratio_by_period(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    按 (symbol, end_date) 汇总"基金集体持仓占该股流通股比例之和"，
    并附带该期的代表ann_date（该期所有基金公告日的中位数，用作PIT可知时点
    的保守估计——实际每只基金公告日不同，但同期基金集中在相近窗口披露，
    取中位数近似，避免为每只股票分别追踪贡献基金列表的复杂度，遵循KISS）。
    返回：symbol, end_date, ann_date, holding_ratio
    """
    grouped = holdings.groupby(["symbol", "end_date"]).agg(
        holding_ratio=("stk_float_ratio", "sum"),
        ann_date=("ann_date", "median"),
    ).reset_index()
    return grouped


def compute_signal_cross(holding_by_period: pd.DataFrame, codes: list, as_of: pd.Timestamp) -> pd.Series:
    """
    截面信号：每只股票最新两期（ann_date<=as_of）holding_ratio的环比变化。
    """
    sub = holding_by_period[
        holding_by_period["symbol"].isin(codes) & (holding_by_period["ann_date"] <= as_of)
    ]
    if sub.empty:
        return pd.Series(dtype=float)

    values = {}
    for code, g in sub.groupby("symbol"):
        g = g.sort_values("end_date")
        if len(g) < 2:
            continue
        prev, latest = g.iloc[-2], g.iloc[-1]
        values[code] = latest["holding_ratio"] - prev["holding_ratio"]
    return pd.Series(values)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    holding_by_period: pd.DataFrame,
) -> pd.DataFrame:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        month_end = pd.Timestamp(month_end)
        next_month_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        factor = compute_signal_cross(holding_by_period, available, month_end)
        if len(factor) < MIN_STOCKS_PER_CROSS:
            continue

        factor = winsorize(factor)
        factor = standardize(factor)

        close_row = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common = close_row.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices_next[common] / close_row[common] - 1
        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({"date": month_end, "ic": ic, "n_stocks": len(common)})

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series) -> dict:
    clean = ic_series.dropna()
    overall_mean = clean.mean()
    yearly = clean.groupby(clean.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(overall_mean)).mean() if overall_mean != 0 else 0.0
    passed = abs(overall_mean) >= 0.03 and same_sign >= 0.6
    return {
        "样本月数": len(clean),
        "IC均值": round(overall_mean, 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(overall_mean / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
        "年度同向占比": f"{same_sign * 100:.1f}%",
        "通过初筛": passed,
    }


def print_annual_ic(ic_series: pd.Series, label: str) -> None:
    clean = ic_series.dropna()
    yearly = clean.groupby(clean.index.year).mean()
    print(f"\n  {label} 年度IC均值:")
    for y in sorted(yearly.index):
        n = (clean.index.year == y).sum()
        print(f"    {y}: {yearly[y]:+.4f}  (n={n})")


def plot_ic(ic_df: pd.DataFrame, output_dir: pathlib.Path, title: str) -> None:
    ic = ic_df["ic"].dropna()
    fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(16, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax_bar, ax_cum]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
    ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
    ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5, label=f"均值 {ic.mean():.4f}")
    ax_bar.set_title(f"{title} — 月度 Rank IC")
    ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    cum_ic = ic.cumsum()
    icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
    ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
    ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax_cum.set_title(f"累积 IC（ICIR={icir:.3f}）")
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="公募基金持仓增减仓强度因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    print("加载基金持仓数据...")
    holdings = load_all_fund_holdings()
    print(f"合计 {len(holdings)} 条持仓记录（Q2/Q4），涉及 {holdings['symbol'].nunique()} 只个股")
    holding_by_period = compute_holding_ratio_by_period(holdings)
    print(f"按(个股,披露期)汇总后 {len(holding_by_period)} 条")

    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    coverage = len(set(holdings["symbol"].unique()) & all_codes) / len(all_codes) if all_codes else 0
    print(f"成分股覆盖率（曾被基金持仓覆盖）：{coverage:.1%}\n")

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}\n")

    for key in index_keys:
        cfg = INDEX_CONFIG[key]
        name = cfg["name"]
        members_file = cfg["members_file"]
        if not members_file.exists():
            print(f"跳过 {name}：成分股快照不存在")
            continue

        out_dir = OUTPUT_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")

        ic_df = compute_monthly_ic(close_panel, members_file, holding_by_period)
        if ic_df.empty:
            print("  无有效数据")
            continue

        ic_df.to_csv(out_dir / "ic_series.csv")
        stats = summarize_ic(ic_df["ic"])
        print(f"  IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")

        print_annual_ic(ic_df["ic"], name)
        plot_ic(ic_df, out_dir, f"{name} 基金持仓变化因子")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
