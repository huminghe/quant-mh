"""
股息率因子截面IC验证

来源：MSCI《中国A股因子表现》2025年12月研报，A股高股息率因子持续跑赢，
与低波动因子并列近年最强因子，动量因子反而最弱（与全球市场排序相反）。
项目此前六轮40+方向+第七/八轮候选中从未测试过该因子，详见
a_stock/docs/research_index_enhancement.md「指数增强策略」章节新候选因子调研（2026-08-17）。

因子来源：daily_basic `dv_ratio`（近12个月现金分红/最新总市值，tushare原生
字段），月度快照直接取自 valuation_monthly.parquet，不需要额外拉取或滚动
窗口计算（比换手率/Amihud等日频构造因子更简单）。

方向：正向（股息率越高，下月收益越高，对齐MSCI文献）。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目既有
factor_ic_*.py方法论。入选阈值：|IC均值|>=0.03 且年度同向占比>=60%。

用法：
  cd a_stock/backtest
  python factor_ic_dividend.py               # 默认跑全部
  python factor_ic_dividend.py --index hs300
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

VALUATION_FILE = DATA_DIR / "valuation_monthly.parquet"
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_dividend"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50
FACTOR_DIRECTION = +1  # 股息率越高，方向为正

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


# ── 数据加载 ──────────────────────────────────────────────

def load_dividend_panel() -> pd.DataFrame:
    """股息率月度快照，宽格式（index=trade_date月末近似日，columns=ts_code）"""
    df = pd.read_parquet(VALUATION_FILE)[["trade_date", "ts_code", "dv_ratio"]].dropna()
    return df.pivot(index="trade_date", columns="ts_code", values="dv_ratio").sort_index()


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    dividend_panel: pd.DataFrame,
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

        close_row = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common_ret = close_row.index.intersection(fwd_prices_next.index)
        if len(common_ret) < MIN_STOCKS_PER_CROSS:
            continue
        fwd_ret = fwd_prices_next[common_ret] / close_row[common_ret] - 1

        pit_snap = dividend_panel.loc[:month_end]
        if pit_snap.empty:
            continue
        cols = [c for c in available if c in dividend_panel.columns]
        factor = pit_snap.iloc[-1][cols].dropna()
        if len(factor) < MIN_STOCKS_PER_CROSS:
            continue

        factor = factor * FACTOR_DIRECTION
        factor = winsorize(factor)
        factor = standardize(factor)

        common = factor.index.intersection(common_ret)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        ic = cross_section_rank_ic(factor[common], fwd_ret[common])
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


ROLLING_WINDOW_MONTHS = 36
WEAK_THRESHOLD = 0.02
REVERSAL_MONTHS = 12


def rolling_ic_analysis(ic_series: pd.Series, label: str) -> dict:
    """
    滚动36个月IC均值分析，判定信号是否持续走弱/反转。
    判定标准（已获用户批准的计划）：
      - 若最近36个月滚动IC均值绝对值<0.02且未反转 → 信号存在但强度不足，暂不接入
      - 若转负且持续超12个月 → 证伪
    """
    clean = ic_series.dropna().sort_index()
    if len(clean) < ROLLING_WINDOW_MONTHS:
        print(f"\n  {label} 滚动IC：样本月数{len(clean)}不足{ROLLING_WINDOW_MONTHS}个月，跳过滚动分析")
        return {}

    rolling = clean.rolling(ROLLING_WINDOW_MONTHS).mean().dropna()
    latest = rolling.iloc[-1]

    # 判断信号是否已转负并持续超过REVERSAL_MONTHS个月（用逐月IC而非滚动均值判断"持续"）
    recent = clean.iloc[-REVERSAL_MONTHS:]
    persistent_negative = (recent < 0).sum() >= REVERSAL_MONTHS * 0.8  # 80%以上月份为负视为持续转负

    if persistent_negative and latest < 0:
        verdict = "证伪：近期信号持续转负"
    elif abs(latest) < WEAK_THRESHOLD:
        verdict = "信号存在但强度不足，暂不接入"
    else:
        verdict = "信号仍稳健"

    print(f"\n  {label} 滚动{ROLLING_WINDOW_MONTHS}个月IC均值:")
    print(f"    最新滚动IC均值 = {latest:+.4f}")
    print(f"    滚动IC均值区间 = [{rolling.min():+.4f}, {rolling.max():+.4f}]")
    print(f"    最近{REVERSAL_MONTHS}个月IC为负的占比 = {(recent < 0).mean():.1%}")
    print(f"    判定：{verdict}")

    return {
        "最新滚动IC均值": round(latest, 4),
        "滚动IC区间": (round(rolling.min(), 4), round(rolling.max(), 4)),
        f"最近{REVERSAL_MONTHS}月IC为负占比": f"{(recent < 0).mean() * 100:.1f}%",
        "判定": verdict,
    }


def plot_rolling_ic(ic_series: pd.Series, output_dir: pathlib.Path, title: str) -> None:
    clean = ic_series.dropna().sort_index()
    rolling = clean.rolling(ROLLING_WINDOW_MONTHS).mean().dropna()
    if rolling.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
        lbl.set_color("white")
    ax.plot(rolling.index, rolling.values, color="#60a5fa", linewidth=1.5)
    ax.fill_between(rolling.index, rolling.values, alpha=0.2, color="#60a5fa")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.axhline(WEAK_THRESHOLD, color="#facc15", linewidth=0.8, linestyle=":", label=f"弱信号阈值±{WEAK_THRESHOLD}")
    ax.axhline(-WEAK_THRESHOLD, color="#facc15", linewidth=0.8, linestyle=":")
    ax.set_title(f"{title} — 滚动{ROLLING_WINDOW_MONTHS}个月IC均值")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout(pad=2.0)
    out_path = output_dir / "rolling_ic.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"滚动IC图已保存：{out_path}")


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
    parser = argparse.ArgumentParser(description="股息率因子截面IC验证")
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

    print("加载股息率月度快照...")
    dividend_panel = load_dividend_panel()

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

        ic_df = compute_monthly_ic(close_panel, dividend_panel, members_file)
        if ic_df.empty:
            print("  无有效数据")
            continue

        ic_df.to_csv(out_dir / "ic_series.csv")
        stats = summarize_ic(ic_df["ic"])
        print(f"  IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")

        print_annual_ic(ic_df["ic"], name)
        plot_ic(ic_df, out_dir, f"{name} 股息率因子")
        rolling_ic_analysis(ic_df["ic"], name)
        plot_rolling_ic(ic_df["ic"], out_dir, f"{name} 股息率因子")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
