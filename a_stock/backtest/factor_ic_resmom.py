"""
动量因子截面 IC 验证（skip-month 动量 + 残差动量）
- skip-month 动量：过去 N 日累积收益，跳过最近 21 日（规避短期反转）
- 残差动量：对市场等权均值做 OLS 后的残差累积收益（LOO 方式剔除自身）
- 验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率）
- 因子方向：越高越好（正向，IC 预期为正）

注意：A 股整体动量效应弱（反转市场），该脚本用于确认是否存在任何形式的动量信号。

用法：
  cd a_stock/backtest
  python factor_ic_resmom.py               # 默认跑全部
  python factor_ic_resmom.py --index hs300
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

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_resmom"

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

# 动量窗口配置（全部用简单累积收益，skip 控制跳过最近 N 日）
FACTOR_CONFIG = {
    "mom_63":       {"name": "动量（63日）",           "window": 63,  "skip": 0},
    "mom_126s21":   {"name": "动量（126日 skip21）",   "window": 126, "skip": 21},
    "mom_252s21":   {"name": "动量（252日 skip21）",   "window": 252, "skip": 21},
}


# ── 工具函数 ──────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
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


# ── 动量计算 ──────────────────────────────────────────────

def compute_momentum(
    close_panel: pd.DataFrame,
    month_end: pd.Timestamp,
    window: int,
    skip: int = 0,
) -> pd.Series:
    """
    简单累积收益动量。
    skip > 0 时跳过最近 skip 日（规避短期反转）：
      取 [-(window+skip+1) : -(skip+1)] 区间的累积收益。
    返回 Series，index=ts_code。
    """
    hist = close_panel.loc[:month_end]
    if len(hist) < window + skip + 2:
        return pd.Series(dtype=float)

    price_end   = hist.iloc[-(skip + 1)]        # skip 日前的收盘价
    price_start = hist.iloc[-(window + skip + 1)]  # 再往前 window 日
    ret = price_end / price_start - 1
    return ret.dropna()


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    factor_key: str,
    members_file: pathlib.Path,
) -> pd.DataFrame:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    cfg = FACTOR_CONFIG[factor_key]

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_month_end = monthly_last[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        # 用收盘价面板计算简单累积收益动量
        close_sub = close_panel[available]
        factor = compute_momentum(
            close_sub, month_end,
            window=cfg["window"], skip=cfg["skip"]
        )
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
        records.append({
            "date":         month_end,
            "ic":           ic,
            "n_stocks":     len(common),
            "factor_mean":  factor.mean(),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    cfg   = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    return {
        "因子":       cfg["name"],
        "样本月数":    len(clean),
        "IC均值":     round(clean.mean(), 4),
        "IC标准差":   round(clean.std(),  4),
        "ICIR":      round(clean.mean() / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比":   f"{(clean > 0).mean() * 100:.1f}%",
        "|IC|>0.02占比": f"{(clean.abs() > 0.02).mean() * 100:.1f}%",
    }


# ── 画图 ──────────────────────────────────────────────────

def plot_ic_results(ic_results: dict, output_dir: pathlib.Path,
                    title_prefix: str = "") -> None:
    n = len(ic_results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (fkey, df) in enumerate(ic_results.items()):
        ax_bar = axes[row_idx][0]
        ax_cum = axes[row_idx][1]
        ic     = df["ic"].dropna()
        fname  = FACTOR_CONFIG[fkey]["name"]

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
        icir   = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
        ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
        ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_cum.set_title(f"{fname} — 累积 IC（ICIR={icir:.3f}）")
        ax_cum.set_ylabel("累积 Rank IC")
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame,
                  factor_keys: list[str]) -> pd.DataFrame:
    cfg          = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_results   = {}
    summary_rows = []

    for fkey in factor_keys:
        fname = FACTOR_CONFIG[fkey]["name"]
        print(f"  [{index_name}] {fname}...")
        ic_df = compute_monthly_ic(close_panel, fkey, members_file)
        if ic_df.empty:
            print(f"    无有效数据")
            continue
        ic_results[fkey] = ic_df

        stats = summarize_ic(ic_df["ic"], fkey)
        summary_rows.append(stats)
        print(f"    IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  n={stats['样本月数']}月")

        ic_df.to_csv(out_dir / f"ic_{fkey}.csv")

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).set_index("因子")
    summary_df.to_csv(out_dir / "ic_summary.csv")

    if ic_results:
        plot_ic_results(ic_results, out_dir, title_prefix=index_name)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="残差动量因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"],
                        default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys  = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]
    factor_keys = list(FACTOR_CONFIG.keys())

    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）\n")

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, factor_keys)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（残差动量因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("解读：")
        print("  IC > 0 = 残差动量越强的股下月涨更多（动量效应成立）")
        print("  |ICIR| > 0.3 显著，> 0.5 强显著")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
