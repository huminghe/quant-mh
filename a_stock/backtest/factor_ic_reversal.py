"""
反转因子截面 IC 验证（多指数 × 多窗口）
- 因子定义：过去 N 日累计收益率，反向使用
- 验证方式：每月末计算截面 Spearman IC（因子值 vs 下月收益率）
- point-in-time：用月末前历史数据计算因子，预测下月收益
- 支持多指数：沪深300 / 中证500
- 窗口：1/5/20 日（短期）+ 21/63 日（中期，1/3个月）
- 输出：IC 汇总表、IC 时序图、5分组累积收益图

用法：
  cd a_stock/backtest
  python factor_ic_reversal.py               # 默认跑全部（两个指数 × 5个窗口）
  python factor_ic_reversal.py --index hs300 # 只跑沪深300
  python factor_ic_reversal.py --index hs500 # 只跑中证500
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

# 加载数据模块
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR

# ── 指数配置 ──────────────────────────────────────────────
INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
        "min_stocks": 50,
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
        "min_stocks": 80,
    },
}

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_reversal"

# 反转窗口：短期 1/5/20 日 + 中期 21/63 日
REVERSAL_WINDOWS = [1, 5, 20, 21, 63]

# 回测区间
START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

# 最小有效股票数（截面 IC 计算要求）
MIN_STOCKS_PER_CROSS = 50


# ── 工具函数 ──────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    """截面去极值：上下各 pct 分位数截断"""
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    """截面 Z-score 标准化"""
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def cross_section_rank_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    """
    计算截面 Spearman 秩相关（Rank IC）。
    factor 和 fwd_ret 按股票代码对齐，去掉 NaN 后计算。
    """
    aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret"]
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
    return ic


# ── 核心计算 ──────────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    window: int,
    members_file: pathlib.Path,
) -> pd.DataFrame:
    """
    按月末截面计算反转因子的 Rank IC。

    参数：
      close_panel: 收盘价面板，index=trade_date，columns=ts_code
      window: 反转因子回望窗口（交易日）
      members_file: 成分股快照 parquet 路径

    返回：
      DataFrame，index=月末日期，columns=['ic', 'n_stocks', 'factor_mean']
    """
    close_panel = close_panel.loc[START_DATE:END_DATE]

    # 月末实际交易日：对每个自然月末，找 <= 月末的最后一个交易日
    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):  # 最后一月无下月数据
        next_month_end = monthly_last[i + 1]

        # ── Point-in-time 成分股 ────────────────────────
        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue

        # 只取在面板中有数据的成分股
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        # ── 反转因子计算 ──────────────────────────────
        # 取月末前 window+1 个交易日的价格窗口（不含当日以模拟当日收盘买入）
        # 因子值 = 过去 window 日累计收益（反转：越低未来收益越高）
        hist = close_panel[available].loc[:month_end]
        if len(hist) < window + 1:
            continue

        # 过去 window 日累计收益率
        factor = hist.iloc[-1] / hist.iloc[-(window + 1)] - 1
        factor = factor.dropna()

        # 去极值 + 标准化
        factor = winsorize(factor)
        factor = standardize(factor)

        # ── 下月收益率 ────────────────────────────────
        # 下月收益：month_end 收盘 → next_month_end 收盘
        fwd_prices_now  = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()

        common = (fwd_prices_now.index
                  .intersection(fwd_prices_next.index)
                  .intersection(factor.index))
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices_next[common] / fwd_prices_now[common] - 1

        # ── 截面 Rank IC ─────────────────────────────
        ic = cross_section_rank_ic(factor[common], fwd_ret)

        records.append({
            "date": month_end,
            "ic": ic,
            "n_stocks": len(common),
            "factor_mean": factor.mean(),
        })

    result = pd.DataFrame(records).set_index("date")
    return result


def compute_quintile_returns(
    close_panel: pd.DataFrame,
    window: int,
    members_file: pathlib.Path,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    计算因子分组的月度平均收益率。
    返回：index=月末日期，columns=分组标签（Q1~Q5）
    """
    close_panel = close_panel.loc[START_DATE:END_DATE]
    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    group_records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_month_end = monthly_last[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < n_groups * 10:
            continue

        hist = close_panel[available].loc[:month_end]
        if len(hist) < window + 1:
            continue

        factor = hist.iloc[-1] / hist.iloc[-(window + 1)] - 1
        factor = factor.dropna()
        factor = winsorize(factor)

        fwd_prices_now  = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common = fwd_prices_now.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < n_groups * 10:
            continue

        fwd_ret = fwd_prices_next[common] / fwd_prices_now[common] - 1

        # 按因子值分 n 组（Q1=因子值最低=跌最多→反转预期最高）
        labels = [f"Q{j+1}" for j in range(n_groups)]
        groups = pd.qcut(factor[common], q=n_groups, labels=labels)

        row = {"date": month_end}
        for label in labels:
            mask = groups == label
            row[label] = fwd_ret[mask].mean()
        group_records.append(row)

    return pd.DataFrame(group_records).set_index("date")


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, window: int) -> dict:
    clean = ic_series.dropna()
    return {
        "因子窗口": f"{window}日反转",
        "样本月数": len(clean),
        "IC均值": round(clean.mean(), 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(clean.mean() / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
        "|IC|>0.02占比": f"{(clean.abs() > 0.02).mean() * 100:.1f}%",
        "IC最大值": round(clean.max(), 4),
        "IC最小值": round(clean.min(), 4),
    }


# ── 画图 ──────────────────────────────────────────────────

def plot_ic_series(ic_results: dict, output_dir: pathlib.Path,
                   title_prefix: str = "") -> None:
    """画 IC 时间序列和累积 IC 图"""
    fig, axes = plt.subplots(len(ic_results), 2, figsize=(16, 4 * len(ic_results)))
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (window, df) in enumerate(ic_results.items()):
        ic = df["ic"].dropna()
        ax_bar  = axes[row_idx, 0] if len(ic_results) > 1 else axes[0]
        ax_cum  = axes[row_idx, 1] if len(ic_results) > 1 else axes[1]

        for ax in [ax_bar, ax_cum]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        # IC 柱状图
        colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
        ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
        ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5,
                       linestyle="-", label=f"均值 {ic.mean():.4f}")
        ax_bar.set_title(f"{title_prefix} {window}日反转 — 月度 Rank IC".strip())
        ax_bar.set_ylabel("Rank IC")
        ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
        ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # 累积 IC
        cum_ic = ic.cumsum()
        ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
        ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
        ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        ax_cum.set_title(f"{window}日反转 — 累积 IC（ICIR={icir:.3f}）")
        ax_cum.set_ylabel("累积 Rank IC")
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


def plot_quintile_returns(quintile_results: dict, output_dir: pathlib.Path,
                          title_prefix: str = "") -> None:
    """画分组累积收益图"""
    n = len(quintile_results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    colors_q = ["#ef4444", "#f97316", "#a3a3a3", "#34d399", "#22c55e"]

    for ax, (window, df) in zip(axes, quintile_results.items()):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")

        cum = (1 + df.fillna(0)).cumprod()
        for j, col in enumerate(df.columns):
            ax.plot(cum.index, cum[col], label=col,
                    color=colors_q[j], linewidth=1.5)
        ax.set_title(f"{title_prefix} {window}日反转 — 分5组累积净值\n（Q1=最强反转预期，Q5=最弱）".strip())
        ax.set_ylabel("累积净值")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "quintile_returns.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"分组收益图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame) -> pd.DataFrame:
    """对单个指数跑所有窗口的 IC 验证，返回汇总 DataFrame"""
    cfg = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在（{members_file}）")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_results = {}
    quintile_results = {}
    summary_rows = []

    for window in REVERSAL_WINDOWS:
        print(f"  [{index_name}] {window}日反转 IC...")
        ic_df = compute_monthly_ic(close_panel, window, members_file)
        ic_results[window] = ic_df

        stats = summarize_ic(ic_df["ic"], window)
        summary_rows.append(stats)
        print(f"    IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  n={stats['样本月数']}月")

        qret = compute_quintile_returns(close_panel, window, members_file)
        quintile_results[window] = qret

    summary_df = pd.DataFrame(summary_rows).set_index("因子窗口")

    # 保存
    summary_df.to_csv(out_dir / "ic_summary.csv")
    for window, df in ic_results.items():
        df.to_csv(out_dir / f"ic_{window}d.csv")

    # 画图
    plot_ic_series(ic_results, out_dir, title_prefix=index_name)
    plot_quintile_returns(quintile_results, out_dir, title_prefix=index_name)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="反转因子截面IC验证（多指数×多窗口）")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"],
                        default="all", help="指定指数（默认 all）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 确定要跑哪些指数
    if args.index == "all":
        index_keys = list(INDEX_CONFIG.keys())
    else:
        index_keys = [args.index]

    # 收集所有指数需要的股票代码，一次性加载面板
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

    # 逐指数跑
    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel)
        if not summary.empty:
            all_summaries[name] = summary

    # 合并汇总打印
    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（反转因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("解读：")
        print("  IC < 0 = 跌多的股下月涨更多（反转效应成立）")
        print("  |ICIR| > 0.3 显著，> 0.5 强显著")
        print("  IC>0 占比越低，反转方向越稳定")

    print(f"\n所有输出在：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
