"""
SUE（标准化盈余惊喜）因子截面IC验证

来源：Li/Liu/Liu/Wei《Replicating and Digesting Anomalies in the Chinese
A-Share Market》(Management Science, 2023) 候选清单第5项，详见
a_stock/docs/research_index_enhancement.md「指数增强策略」章节因子候选清单。

因子定义：SUE = (本季单季度EPS - 去年同季单季度EPS) / 近8个单季度差值的标准差。
`financials/{ts_code}.parquet` 里的 eps 是当年累计值（YTD），不是单季度值，
需要先按 end_date 做季度差分还原单季度EPS：
  Q1 EPS = 累计EPS(Q1)
  Q2 EPS = 累计EPS(H1) - 累计EPS(Q1)
  Q3 EPS = 累计EPS(9M) - 累计EPS(H1)
  Q4 EPS = 累计EPS(FY) - 累计EPS(9M)
point-in-time：用 ann_date 作为可知时点，只用已公告的季度。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目既有
factor_ic_*.py方法论。入选阈值：|IC均值|>=0.03 且年度同向占比>=60%。

用法：
  cd a_stock/backtest
  python factor_ic_sue.py               # 默认跑全部
  python factor_ic_sue.py --index hs300
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
from fetch_financials import load_financials

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_sue"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50
MIN_QUARTERS = 8  # 计算SUE标准差所需的最少历史单季度差值数

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


# ── 单季度EPS还原 + SUE计算 ────────────────────────────────

_single_q_cache: dict[str, pd.DataFrame] = {}


def get_single_quarter_eps(ts_code: str) -> pd.DataFrame:
    """
    还原单季度EPS：按自然年内 Q1/Q2/Q3/Q4 顺序差分累计值。
    跨年（Q1）不减，年内(Q2/Q3/Q4)减上一条记录的累计EPS。
    返回列：end_date, ann_date, eps_ytd, eps_q（单季度）
    """
    if ts_code in _single_q_cache:
        return _single_q_cache[ts_code]

    df = load_financials(ts_code)
    if df.empty or "eps" not in df.columns:
        _single_q_cache[ts_code] = pd.DataFrame()
        return pd.DataFrame()

    df = df.sort_values("end_date").drop_duplicates(subset="end_date", keep="last").reset_index(drop=True)
    df["quarter"] = df["end_date"].dt.quarter
    df["year"] = df["end_date"].dt.year

    eps_q = []
    for i, row in df.iterrows():
        if row["quarter"] == 1:
            eps_q.append(row["eps"])
        else:
            prev = df[(df["year"] == row["year"]) & (df["quarter"] == row["quarter"] - 1)]
            if prev.empty or pd.isna(prev.iloc[0]["eps"]) or pd.isna(row["eps"]):
                eps_q.append(np.nan)
            else:
                eps_q.append(row["eps"] - prev.iloc[0]["eps"])
    df["eps_q"] = eps_q

    result = df[["end_date", "ann_date", "eps", "eps_q", "quarter", "year"]].rename(
        columns={"eps": "eps_ytd"}
    )
    _single_q_cache[ts_code] = result
    return result


def compute_sue_pit(ts_code: str, as_of: pd.Timestamp) -> float | None:
    """
    PIT计算单只股票的SUE：只用 ann_date <= as_of 的季度记录。
    最新季度 vs 去年同季度的单季度EPS差值，除以近MIN_QUARTERS个单季度差值的标准差。
    """
    df = get_single_quarter_eps(ts_code)
    if df.empty:
        return None

    valid = df[df["ann_date"] <= as_of].dropna(subset=["eps_q"]).reset_index(drop=True)
    if len(valid) < MIN_QUARTERS + 4:
        return None

    latest = valid.iloc[-1]
    same_q_last_year = valid[
        (valid["year"] == latest["year"] - 1) & (valid["quarter"] == latest["quarter"])
    ]
    if same_q_last_year.empty:
        return None

    surprise = latest["eps_q"] - same_q_last_year.iloc[0]["eps_q"]

    # 近8个单季度同比差值（含本期），用于标准化尺度
    hist_q = valid.iloc[-MIN_QUARTERS:]
    diffs = []
    for _, row in hist_q.iterrows():
        prior = valid[(valid["year"] == row["year"] - 1) & (valid["quarter"] == row["quarter"])]
        if not prior.empty:
            diffs.append(row["eps_q"] - prior.iloc[0]["eps_q"])
    if len(diffs) < MIN_QUARTERS - 2:
        return None

    std = np.std(diffs, ddof=1)
    if std < 1e-8 or pd.isna(std):
        return None

    return surprise / std


def compute_sue_cross(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        v = compute_sue_pit(code, month_end)
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
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

        factor = compute_sue_cross(available, month_end)
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
    parser = argparse.ArgumentParser(description="SUE（标准化盈余惊喜）因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

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

        ic_df = compute_monthly_ic(close_panel, members_file)
        if ic_df.empty:
            print("  无有效数据")
            continue

        ic_df.to_csv(out_dir / "ic_series.csv")
        stats = summarize_ic(ic_df["ic"])
        print(f"  IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")

        print_annual_ic(ic_df["ic"], name)
        plot_ic(ic_df, out_dir, f"{name} SUE因子")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
