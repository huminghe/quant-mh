"""
MAX因子/彩票偏好 截面IC验证

来源：Bali/Cakici/Whitelaw《Maxing Out》(JFE 2011) 及其A股复现（EFMA 2025
工作论文）。投资者对"小概率暴涨"存在彩票式偏好，过度买入近期出现过极端
高收益日的股票，推高定价随后收益走低（负向使用）。此前项目已测试筹码分布
/获利比例（winner_rate，IC不足证伪）和高送转除权填权（事件研究不显著），
均属"投机偏好定价"大类但机制不同（前者是持仓成本分布，后者是公司行为事件），
MAX因子是纯粹的"过去极端收益记忆"，此前项目从未测试过。详见
a_stock/docs/research_index_enhancement.md「指数增强策略」章节候选清单。

因子构造（仅用stock_daily收盘价，零新增数据成本）：
- MAX(N) = 过去21个交易日（1个月）内最大N日收益的均值，N取1/5两档
  （N=1即Bali原版单日最大收益；N=5做稳健性对照，避免单日极值噪声）
- 负向使用（高MAX -> 彩票偏好推高定价 -> 预期未来收益走低）

风险提示（本脚本执行前已知）：MAX因子和已证伪的ST突显效应因子（同样基于
过去1个月日收益的极值/尾部信息）在方法论上高度相似——ST证伪的关键原因是
组合层面月度截面排序剧烈波动导致月换手率77-78%，成本侵蚀了IC阶段的微弱
优势。本脚本只做IC初筛（低成本诊断），不直接上组合回测，IC初筛通过后再
评估是否值得为高换手风险投入完整组合验证。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目
既有factor_ic_*.py方法论（沪深300/中证500成分股，月度截面）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%（项目既定阈值）。

用法：
  cd a_stock/backtest
  python factor_ic_max.py               # 默认跑全部
  python factor_ic_max.py --index hs300
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

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_max"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50
MAX_WINDOW = 21  # 过去21个交易日（1个月）

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

FACTOR_CONFIG = {
    "max1": {"name": "MAX(1)（21日内最大单日收益）",     "direction": -1, "top_n": 1},
    "max5": {"name": "MAX(5)（21日内最大5日收益均值）",   "direction": -1, "top_n": 5},
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


# ── MAX因子计算 ───────────────────────────────────────────

def compute_max_factor(daily_ret_window: pd.DataFrame, top_n: int) -> pd.Series:
    """
    对给定窗口（index=trade_date, columns=ts_code的日收益面板）逐股票取
    最大top_n日收益的均值。返回Series，index=ts_code。
    """
    def _top_mean(col: pd.Series) -> float:
        vals = col.dropna()
        if len(vals) < MAX_WINDOW // 2:
            return np.nan
        return vals.nlargest(top_n).mean()
    return daily_ret_window.apply(_top_mean, axis=0)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    close_panel = close_panel.loc[START_DATE:END_DATE]
    daily_ret = close_panel.pct_change()

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = {fkey: [] for fkey in FACTOR_CONFIG}

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

        hist = daily_ret[available].loc[:month_end]
        if len(hist) < MAX_WINDOW + 1:
            continue
        window = hist.iloc[-MAX_WINDOW:]

        for fkey, cfg in FACTOR_CONFIG.items():
            factor = compute_max_factor(window, cfg["top_n"]).dropna()
            if len(factor) < MIN_STOCKS_PER_CROSS:
                continue

            factor = factor * cfg["direction"]
            factor = winsorize(factor)
            factor = standardize(factor)

            common = factor.index.intersection(common_ret)
            if len(common) < MIN_STOCKS_PER_CROSS:
                continue

            ic = cross_section_rank_ic(factor[common], fwd_ret[common])
            records[fkey].append({"date": month_end, "ic": ic, "n_stocks": len(common)})

    return {
        fkey: (pd.DataFrame(recs).set_index("date") if recs else pd.DataFrame())
        for fkey, recs in records.items()
    }


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    cfg = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    overall_mean = clean.mean()
    yearly = clean.groupby(clean.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(overall_mean)).mean() if overall_mean != 0 else 0.0
    passed = abs(overall_mean) >= 0.03 and same_sign >= 0.6
    return {
        "因子": cfg["name"],
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


def plot_ic_results(ic_results: dict, output_dir: pathlib.Path, title_prefix: str = "") -> None:
    n = len(ic_results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (fkey, df) in enumerate(ic_results.items()):
        ax_bar = axes[row_idx][0]
        ax_cum = axes[row_idx][1]
        ic = df["ic"].dropna()
        fname = FACTOR_CONFIG[fkey]["name"]

        for ax in [ax_bar, ax_cum]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
        ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
        ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5,
                       linestyle="-", label=f"均值 {ic.mean():.4f}")
        ax_bar.set_title(f"{title_prefix} {fname} — 月度 Rank IC".strip())
        ax_bar.set_ylabel("Rank IC")
        ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
        ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        cum_ic = ic.cumsum()
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
        ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
        ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_cum.set_title(f"{fname} — 累积 IC（ICIR={icir:.3f}）")
        ax_cum.set_ylabel("累积 Rank IC")
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame) -> pd.DataFrame:
    cfg = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{index_name}] 计算月度截面IC...")
    ic_results = compute_monthly_ic(close_panel, members_file)

    summary_rows = []
    for fkey, ic_df in ic_results.items():
        if ic_df.empty:
            print(f"    {FACTOR_CONFIG[fkey]['name']}：无有效数据")
            continue
        stats = summarize_ic(ic_df["ic"], fkey)
        summary_rows.append(stats)
        print(f"    {stats['因子']:<30} IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")
        print_annual_ic(ic_df["ic"], stats["因子"])
        ic_df.to_csv(out_dir / f"ic_{fkey}.csv")

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).set_index("因子")
    summary_df.to_csv(out_dir / "ic_summary.csv")

    valid_results = {k: v for k, v in ic_results.items() if not v.empty}
    if valid_results:
        plot_ic_results(valid_results, out_dir, title_prefix=index_name)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="MAX因子/彩票偏好 截面IC验证")
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
    all_codes = list(all_codes)

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"\n{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（MAX因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("判定标准：|IC均值|>=0.03 且年度同向占比>=60% 为通过初筛")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
