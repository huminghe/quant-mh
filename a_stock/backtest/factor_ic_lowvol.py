"""
低波动因子截面 IC 验证（特质波动率 + 总波动率对比）
- 特质波动率：过去63日日收益率对市场指数回归后的残差标准差（剔除 Beta 暴露）
- 总波动率：过去21/63日收益率标准差（作为对比基准）
- 验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率）
- 因子方向：越低越好（低波动溢价），IC 预期为负
- 支持多指数：沪深300 / 中证500

用法：
  cd a_stock/backtest
  python factor_ic_lowvol.py               # 默认跑全部
  python factor_ic_lowvol.py --index hs300
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
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_lowvol"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50

INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
        "index_code": "000300.SH",
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
        "index_code": "000905.SH",
    },
}

# 要验证的因子
FACTOR_CONFIG = {
    "ivol_63": {
        "name": "特质波动率（63日）",
        "direction": -1,   # 越低越好
    },
    "tvol_21": {
        "name": "总波动率（21日）",
        "direction": -1,
    },
    "tvol_63": {
        "name": "总波动率（63日）",
        "direction": -1,
    },
}

# 市场指数日线数据（从个股面板中取，用于回归）
# hs300 → 000300.SH, hs500 → 000905.SH
# tushare 股票代码格式：600000.SH
# 但指数不在 stock_daily/ 里，改用全体成分股等权作为市场代理
# （用全成分股等权均值替代指数，避免额外拉指数数据）


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


# ── 因子计算 ──────────────────────────────────────────────

def compute_idiosyncratic_vol(
    ret_panel: pd.DataFrame,  # 日收益率面板，index=date，columns=ts_code
    month_end: pd.Timestamp,
    window: int = 63,
    min_obs: int = 30,
) -> pd.Series:
    """
    特质波动率：对每只股票，用过去 window 日日收益率对成分股等权市场收益率做 OLS，
    取残差的年化标准差（× sqrt(252)）。
    返回 Series，index=ts_code。
    """
    # 取 month_end 前 window 日（不含当日）
    hist = ret_panel.loc[:month_end].iloc[-(window + 1):-1]
    if len(hist) < min_obs:
        return pd.Series(dtype=float)

    # 市场代理：所有成分股等权均值
    market_ret = hist.mean(axis=1)

    # 预计算各列的等权总和，用于 leave-one-out 市场代理
    n_valid_per_day = hist.notna().sum(axis=1)
    col_sum = hist.sum(axis=1)

    ivol = {}
    for code in hist.columns:
        stock_ret = hist[code].dropna()
        if len(stock_ret) < min_obs:
            continue
        # leave-one-out 市场代理：排除自身，避免残差偏小
        n_valid = n_valid_per_day.loc[stock_ret.index]
        loo_market = (col_sum.loc[stock_ret.index] - stock_ret) / (n_valid - 1).clip(lower=1)
        X = np.column_stack([np.ones(len(loo_market)), loo_market.values])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, stock_ret.values, rcond=None)
            resid = stock_ret.values - X @ beta
            ivol[code] = resid.std() * np.sqrt(252)
        except Exception:
            continue

    return pd.Series(ivol)


def compute_total_vol(
    ret_panel: pd.DataFrame,
    month_end: pd.Timestamp,
    window: int,
    min_obs: int = 15,
) -> pd.Series:
    """总波动率：过去 window 日日收益率的年化标准差"""
    hist = ret_panel.loc[:month_end].iloc[-(window + 1):-1]
    if len(hist) < min_obs:
        return pd.Series(dtype=float)
    return hist.std() * np.sqrt(252)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    factor_key: str,
    members_file: pathlib.Path,
) -> pd.DataFrame:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    # 日收益率面板（提前算好，避免重复计算）
    ret_panel = close_panel.pct_change()

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_month_end = monthly_last[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        # 只取当期成分股的收益率面板
        ret_sub = ret_panel[available]

        # 计算因子值
        if factor_key == "ivol_63":
            factor = compute_idiosyncratic_vol(ret_sub, month_end, window=63)
        elif factor_key == "tvol_21":
            factor = compute_total_vol(ret_sub, month_end, window=21)
        elif factor_key == "tvol_63":
            factor = compute_total_vol(ret_sub, month_end, window=63)
        else:
            continue

        if len(factor) < MIN_STOCKS_PER_CROSS:
            continue

        # 方向调整（越低越好 → 取负后正向使用）
        factor = factor * FACTOR_CONFIG[factor_key]["direction"]

        factor = winsorize(factor)
        factor = standardize(factor)

        # 下月收益率
        close_row = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common = close_row.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices_next[common] / close_row[common] - 1

        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({
            "date": month_end,
            "ic": ic,
            "n_stocks": len(common),
            "factor_mean": factor.mean(),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    cfg = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    return {
        "因子": cfg["name"],
        "样本月数": len(clean),
        "IC均值": round(clean.mean(), 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(clean.mean() / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame,
                  factor_keys: list[str]) -> pd.DataFrame:
    cfg = INDEX_CONFIG[index_key]
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
    parser = argparse.ArgumentParser(description="低波动因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"],
                        default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys  = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]
    factor_keys = list(FACTOR_CONFIG.keys())

    # 收集所有成分股，一次性加载面板
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
        print("全量汇总（低波动因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("解读：")
        print("  因子已取负（方向调整），IC > 0 = 低波动股下月涨更多（低波动溢价成立）")
        print("  |ICIR| > 0.3 显著，> 0.5 强显著")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
