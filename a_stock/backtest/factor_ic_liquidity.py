"""
流动性/交易摩擦类因子截面IC验证（换手率 + Amihud非流动性 + Corwin-Schultz价差）

来源：Li/Liu/Liu/Wei《Replicating and Digesting Anomalies in the Chinese
A-Share Market》(Management Science, 2023)，trading frictions类异象在A股
显著率32.17%，六大类里最强（与美股相反）。此前项目从未测试过该类因子，
详见 a_stock/docs/research.md「指数增强策略」章节候选清单。

三个因子均用现有 stock_daily（high/low/vol/amount）+ valuation_monthly
（circ_mv）数据构造，不新增拉取：
- 换手率：amount / circ_mv 滚动21日均值（circ_mv取最近月末快照，amount为
  日频真实成交额；比值不还原真实流通股数，但保留换手率的截面排序，Rank IC
  只依赖排序不依赖量纲）。方向：高换手率 -> 低预期收益，因子取负后正向使用。
- Amihud非流动性：|日收益率| / amount，滚动126日均值。
  方向：高非流动性 -> 高预期收益溢价，正向使用。
- Corwin-Schultz价差：用连续两日high/low估计隐含买卖价差（Corwin & Schultz
  2012），滚动21日均值，不需要盘口tick数据。方向：价差大 -> 高预期收益溢价，
  正向使用。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目既有
factor_ic_*.py方法论（沪深300/中证500成分股，月度截面）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%（项目既定阈值）。

用法：
  cd a_stock/backtest
  python factor_ic_liquidity.py               # 默认跑全部
  python factor_ic_liquidity.py --index hs300
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

STOCK_DIR = DATA_DIR / "stock_daily"
VALUATION_FILE = DATA_DIR / "valuation_monthly.parquet"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_liquidity"

START_DATE = "2016-01-01"
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

FACTOR_CONFIG = {
    "turnover":  {"name": "换手率（amount/circ_mv，21日均值）",  "direction": -1, "window": 21},
    "amihud":    {"name": "Amihud非流动性（126日均值）",         "direction": +1, "window": 126},
    "cs_spread": {"name": "Corwin-Schultz价差（21日均值）",       "direction": +1, "window": 21},
}

CS_K = 3 - 2 * np.sqrt(2)  # Corwin-Schultz 常数项


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

def load_field_panel(codes: list[str], field: str) -> pd.DataFrame:
    """读取指定字段的宽格式面板（index=trade_date，columns=ts_code），字段取自 stock_daily"""
    frames = {}
    for code in codes:
        path = STOCK_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", field])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames[code] = df.set_index("trade_date")[field]
    if not frames:
        raise FileNotFoundError(f"没有找到任何 {field} 数据")
    return pd.DataFrame(frames).sort_index()


def load_circ_mv_panel() -> pd.DataFrame:
    """流通市值月度快照，宽格式（index=trade_date月末近似日，columns=ts_code）"""
    df = pd.read_parquet(VALUATION_FILE)[["trade_date", "ts_code", "circ_mv"]].dropna()
    return df.pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index()


# ── 因子计算（日频面板，滚动均值后在月末截面取值） ─────────

def compute_turnover_daily(amount_panel: pd.DataFrame, circ_mv_panel: pd.DataFrame) -> pd.DataFrame:
    """换手率代理 = amount / circ_mv（circ_mv 月度快照按日前向填充对齐到日频）"""
    circ_mv_daily = circ_mv_panel.reindex(amount_panel.index, method="ffill")
    common_cols = amount_panel.columns.intersection(circ_mv_daily.columns)
    return amount_panel[common_cols] / circ_mv_daily[common_cols].replace(0, np.nan)


def compute_amihud_daily(close_panel: pd.DataFrame, amount_panel: pd.DataFrame) -> pd.DataFrame:
    """Amihud非流动性 = |日收益率| / amount"""
    common_cols = close_panel.columns.intersection(amount_panel.columns)
    ret = close_panel[common_cols].pct_change().abs()
    return ret / amount_panel[common_cols].replace(0, np.nan)


def compute_cs_spread_daily(high_panel: pd.DataFrame, low_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Corwin-Schultz (2012) 隐含买卖价差估计量，用连续两日 high/low 构造。
    beta：两日对数high/low比值平方和；gamma：两日窗口内max(high)/min(low)对数平方。
    alpha 由 beta/gamma 组合而来，spread = 2*(e^alpha - 1)/(1 + e^alpha)，负值截断为0
    （文献惯例：负值视为估计噪声，无经济意义）。
    """
    common_cols = high_panel.columns.intersection(low_panel.columns)
    high = high_panel[common_cols]
    low = low_panel[common_cols]

    ln_hl = np.log(high / low)
    beta = ln_hl ** 2 + ln_hl.shift(1) ** 2

    high_max = high.combine(high.shift(1), np.maximum)
    low_min = low.combine(low.shift(1), np.minimum)
    gamma = np.log(high_max / low_min) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    factor_panels: dict[str, pd.DataFrame],
    members_file: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = {fkey: [] for fkey in factor_panels}

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
            cols = [c for c in available if c in panel.columns]
            if not cols or month_end not in panel.index:
                continue
            factor = panel[cols].loc[:month_end].iloc[-cfg["window"]:].mean()
            factor = factor.dropna()
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
    """
    年度同向占比：按项目既定口径（同 etf_rotation_v37 的 report_ic），
    是"各年度IC均值与全样本IC均值符号一致"的年份占比，不是月度符号占比。
    """
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

def run_one_index(
    index_key: str,
    close_panel: pd.DataFrame,
    factor_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cfg = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{index_name}] 计算月度截面IC...")
    ic_results = compute_monthly_ic(close_panel, factor_panels, members_file)

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
    parser = argparse.ArgumentParser(description="流动性/交易摩擦类因子截面IC验证")
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

    print(f"加载收盘价/最高价/最低价/成交额面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=all_codes)
    high_panel = load_field_panel(all_codes, "high")
    low_panel = load_field_panel(all_codes, "low")
    amount_panel = load_field_panel(all_codes, "amount")
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    print("加载流通市值月度快照...")
    circ_mv_panel = load_circ_mv_panel()

    print("计算因子面板（换手率/Amihud/Corwin-Schultz价差）...")
    factor_panels = {
        "turnover": compute_turnover_daily(amount_panel, circ_mv_panel),
        "amihud": compute_amihud_daily(close_panel, amount_panel),
        "cs_spread": compute_cs_spread_daily(high_panel, low_panel),
    }

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"\n{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, factor_panels)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（流动性因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("判定标准：|IC均值|>=0.03 且年度同向占比>=60% 为通过初筛")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
