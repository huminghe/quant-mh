"""
突显效应/Salience Theory 因子截面IC验证（ST + CS指标）

来源：Cosemans & Frehen《Salience Theory and Stock Prices: Empirical
Evidence》(JFE 2021)；A股复现：Pacific-Basin Finance Journal 2024。
投资者对个股历史收益"显眼程度"存在认知偏差，显眼的高收益被过度关注推高
定价，随后收益走低。此前项目从未测试过该类因子，详见
a_stock/docs/research.md「指数增强策略」章节候选清单。

因子构造（仅用现有stock_daily收盘价，零新增数据成本）：
- 突显度函数：σ(r_is, r_s) = |r_is - r_s| / (|r_is| + |r_s| + θ)，θ=0.1
  （Bordalo/Gennaioli/Shleifer 2012标定值），r_is为个股日收益，r_s为市场
  （截面等权）日收益，基于过去21个交易日（1个月）窗口
- 突显权重：按σ降序排名，ω_is = δ^(rank-1) / Σδ^(rank-1)，δ<1（本脚本用
  δ=0.7，与原文献ω递减权重一致），rank=1为最显眼
- ST因子：突显加权收益 - 等权收益（ST = Σω_is*r_is - (1/n)Σr_is），负向
  使用（高ST代表被高估显眼的正收益，预期未来收益走低）
- CS因子：ST的月度环比变化（CS_t = ST_t - ST_{t-1}），负向使用（A股复现
  文献用"突显度变化趋势"而非水平值预测）

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目
既有factor_ic_*.py方法论（沪深300/中证500成分股，月度截面）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%（项目既定阈值）。
异质性检验重点：沪深300（机构主导、主板为主）vs 中证500 上ST/CS的IC强度
对照，对应文献"主板显著、创业板不显著，随机构持股比例上升而减弱"的结论。

用法：
  cd a_stock/backtest
  python factor_ic_salience.py               # 默认跑全部
  python factor_ic_salience.py --index hs300
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

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_salience"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50
SALIENCE_WINDOW = 21   # 过去21个交易日（1个月）
THETA = 0.1            # BGS(2012)标定值
DELTA = 0.7            # 突显权重递减系数

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
    "st": {"name": "ST突显效应（突显加权收益-等权收益）", "direction": -1},
    "cs": {"name": "CS突显度变化（ST月度环比）",           "direction": -1},
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


# ── 突显效应因子计算 ──────────────────────────────────────

def compute_salience_weighted_return(daily_ret_window: pd.DataFrame) -> pd.Series:
    """
    对给定窗口（index=trade_date, columns=ts_code的日收益面板）计算单期ST值。
    市场收益r_s取窗口内每日截面等权均值；对每只股票逐日计算突显度σ，按σ
    降序（最显眼=rank1）赋递减权重ω，ST = Σ(ω_is*r_is)/Σω_is - mean(r_is)。
    返回 Series，index=ts_code。
    """
    r_s = daily_ret_window.mean(axis=1)  # 每日市场（截面等权）收益，Series index=trade_date

    st_values = {}
    for code in daily_ret_window.columns:
        r_is = daily_ret_window[code].dropna()
        if len(r_is) < SALIENCE_WINDOW // 2:
            continue
        r_s_aligned = r_s.reindex(r_is.index)
        sigma = (r_is - r_s_aligned).abs() / (r_is.abs() + r_s_aligned.abs() + THETA)
        sigma = sigma.dropna()
        if sigma.empty:
            continue
        r_is = r_is.reindex(sigma.index)

        # 按σ降序排名（rank1=最显眼），赋递减权重 δ^(rank-1)
        rank = sigma.rank(ascending=False, method="first")
        weight = DELTA ** (rank - 1)
        weight = weight / weight.sum()

        st = (weight * r_is).sum() - r_is.mean()
        st_values[code] = st

    return pd.Series(st_values)


def compute_monthly_st_cs(
    close_panel: pd.DataFrame,
    monthly_last: np.ndarray,
) -> pd.DataFrame:
    """
    对每个月末，用过去SALIENCE_WINDOW个交易日计算ST因子截面值；
    CS = 当月ST - 上月ST。返回宽格式DataFrame（index=月末日, columns=ts_code），
    含两层：分别通过 .xs 或直接返回两个DataFrame。这里返回 dict{"st":df, "cs":df}。
    """
    daily_ret = close_panel.pct_change()

    st_rows = {}
    for month_end in monthly_last:
        month_end = pd.Timestamp(month_end)
        hist = daily_ret.loc[:month_end]
        if len(hist) < SALIENCE_WINDOW + 1:
            continue
        window = hist.iloc[-SALIENCE_WINDOW:]
        st_rows[month_end] = compute_salience_weighted_return(window)

    st_df = pd.DataFrame(st_rows).T.sort_index()
    cs_df = st_df.diff()
    return {"st": st_df, "cs": cs_df}


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    print("    计算ST/CS因子面板（逐月突显度加权收益，可能较慢）...")
    factor_panels = compute_monthly_st_cs(close_panel, monthly_last)

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

        for fkey, panel in factor_panels.items():
            cfg = FACTOR_CONFIG[fkey]
            if month_end not in panel.index:
                continue
            factor = panel.loc[month_end]
            cols = [c for c in available if c in factor.index]
            factor = factor[cols].dropna()
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
    """年度同向占比：各年度IC均值与全样本IC均值符号一致的年份占比。"""
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
    parser = argparse.ArgumentParser(description="突显效应/Salience Theory 因子截面IC验证")
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
        print("全量汇总（突显效应因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("判定标准：|IC均值|>=0.03 且年度同向占比>=60% 为通过初筛")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
