"""
策略组合分析脚本
用法：python portfolio_analysis.py <excel1> <excel2> [excel3] ... [--names n1 n2 ...]
输入：TradingView 策略测试器导出的 Excel 文件（支持 2~4 个）
输出：相关性矩阵 + 所有组合的 Sharpe/最大回撤/月胜率对比表

示例：
  python portfolio_analysis.py v3_3h_eth.xlsx v2_eth.xlsx
  python portfolio_analysis.py v3_3h_eth.xlsx v2_eth.xlsx ema_eth.xlsx --names v3_3h v2 ema
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_monthly_pnl(filepath: str) -> pd.Series:
    """从 Excel 交易清单 sheet 提取月度 P&L 序列。"""
    df = pd.read_excel(filepath, sheet_name="交易清单", header=0)
    exits = df[df["类型"].str.contains("出场")].copy()
    exits["日期和时间"] = pd.to_datetime(exits["日期和时间"])
    monthly = (
        exits.groupby(exits["日期和时间"].dt.to_period("M"))["净损益 USDT"]
        .sum()
    )
    # 用文件名（去掉路径和扩展名）作为 Series 名称
    name = Path(filepath).stem
    monthly.name = name
    return monthly


def short_name(full_name: str) -> str:
    """从文件名提取简短策略名，如 v3_3h_strategy_eth_... → v3_3h。"""
    parts = full_name.split("_")
    # 取前两段，通常是 v3+3h 或 v2 或 ema
    if len(parts) >= 2 and parts[1] in ("3h", "strategy"):
        return f"{parts[0]}_{parts[1]}" if parts[1] == "3h" else parts[0]
    return parts[0]


# ── 统计计算 ──────────────────────────────────────────────────────────────────

def portfolio_stats(series: pd.Series) -> dict:
    """计算单个月度 P&L 序列的统计指标。"""
    n = len(series)
    mean = series.mean()
    std = series.std(ddof=1)

    # 年化 Sharpe（无风险利率=0）
    sharpe = (mean / std * np.sqrt(12)) if std > 0 else 0.0

    # 月胜率
    win_rate = (series > 0).sum() / n * 100

    # 最大回撤（从累计净值曲线）
    cumulative = series.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_dd = drawdown.min()

    return {
        "月数": n,
        "累计P&L": round(series.sum(), 0),
        "月均P&L": round(mean, 0),
        "最大回撤": round(max_dd, 0),
        "Sharpe": round(sharpe, 3),
        "月胜率%": round(win_rate, 1),
        "最差月": round(series.min(), 0),
        "最佳月": round(series.max(), 0),
    }


def build_portfolio(series_list: list[pd.Series]) -> pd.Series:
    """等权合并多个月度 P&L 序列（对齐日期后求均值）。"""
    df = pd.concat(series_list, axis=1).dropna()
    return df.mean(axis=1)


# ── 输出格式 ──────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_correlation_matrix(names: list[str], df_aligned: pd.DataFrame):
    print_section("策略相关性矩阵（月度）")
    corr = df_aligned.corr()
    # 表头
    header = f"{'':20}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * (20 + 12 * len(names)))
    for row_name in names:
        row = f"{row_name:20}"
        for col_name in names:
            val = corr.loc[row_name, col_name]
            row += f"{val:>12.3f}"
        print(row)

    # 两两相关性解读
    print()
    for a, b in combinations(names, 2):
        val = corr.loc[a, b]
        if val >= 0.8:
            level = "极高相关，组合分散效果有限"
        elif val >= 0.6:
            level = "高相关，分散效果一般"
        elif val >= 0.4:
            level = "中等相关"
        else:
            level = "低相关，组合分散效果好"
        print(f"  {a} vs {b}：{val:.3f}  →  {level}")


def print_stats_table(results: list[tuple[str, dict]]):
    print_section("组合统计对比")
    cols = ["累计P&L", "最大回撤", "Sharpe", "月胜率%", "最差月", "最佳月", "月均P&L", "月数"]
    header = f"{'组合':28}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * (28 + 12 * len(cols)))

    # 找最优值用于标注
    best_sharpe = max(r[1]["Sharpe"] for r in results)
    best_winrate = max(r[1]["月胜率%"] for r in results)
    best_dd = max(r[1]["最大回撤"] for r in results)  # 最大回撤是负数，max 即绝对值最小

    for name, s in results:
        markers = []
        if s["Sharpe"] == best_sharpe:
            markers.append("★Sharpe")
        if s["月胜率%"] == best_winrate:
            markers.append("★胜率")
        if s["最大回撤"] == best_dd:
            markers.append("★回撤最小")
        label = f"{name} {'  '.join(markers)}"

        row = f"{label:28}"
        for c in cols:
            row += f"{s[c]:>12}"
        print(row)


def print_recommendation(results: list[tuple[str, dict]], n_strategies: int):
    print_section("结论")
    # 按 Sharpe 排序
    ranked = sorted(results, key=lambda x: x[1]["Sharpe"], reverse=True)
    best_name, best_stats = ranked[0]
    second_name, second_stats = ranked[1]

    print(f"最优组合：{best_name}")
    print(f"  Sharpe {best_stats['Sharpe']}，月胜率 {best_stats['月胜率%']}%，"
          f"最大回撤 {best_stats['最大回撤']:,.0f} USDT")

    if n_strategies > 1:
        sharpe_diff = best_stats["Sharpe"] - second_stats["Sharpe"]
        print(f"\n次优组合：{second_name}（Sharpe 差距 {sharpe_diff:.3f}）")

    # 判断组合是否优于最优单策略
    single_results = [(n, s) for n, s in results if "+" not in n]
    if single_results:
        best_single = max(single_results, key=lambda x: x[1]["Sharpe"])
        if "+" in best_name:
            improvement = best_stats["Sharpe"] - best_single[1]["Sharpe"]
            print(f"\n组合 vs 最优单策略（{best_single[0]}，Sharpe {best_single[1]['Sharpe']}）：")
            print(f"  Sharpe 提升 +{improvement:.3f}（{improvement/best_single[1]['Sharpe']*100:.1f}%）")
        else:
            print(f"\n单策略已是最优，组合运行无明显收益。")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # 解析 --names 参数
    args = sys.argv[1:]
    custom_names: list[str] = []
    if "--names" in args:
        idx = args.index("--names")
        custom_names = args[idx + 1:]
        args = args[:idx]

    if len(args) < 2:
        print("用法：python portfolio_analysis.py <excel1> <excel2> [excel3] ... [--names n1 n2 ...]")
        print("示例：python portfolio_analysis.py v3_3h_eth.xlsx v2_eth.xlsx ema_eth.xlsx --names v3_3h v2 ema")
        sys.exit(1)

    filepaths = args
    if len(filepaths) > 4:
        print("最多支持 4 个策略文件")
        sys.exit(1)

    if custom_names and len(custom_names) != len(filepaths):
        print(f"--names 数量（{len(custom_names)}）与文件数量（{len(filepaths)}）不匹配")
        sys.exit(1)

    # 加载数据
    print(f"\n加载 {len(filepaths)} 个策略文件...")
    series_map: dict[str, pd.Series] = {}
    for i, fp in enumerate(filepaths):
        monthly = load_monthly_pnl(fp)
        name = custom_names[i] if custom_names else short_name(monthly.name)
        series_map[name] = monthly
        date_range = f"{monthly.index.min()} ~ {monthly.index.max()}"
        print(f"  {name:20} {len(monthly):3d} 个月  {date_range}")

    names = list(series_map.keys())

    # 对齐所有序列（取交集）
    df_aligned = pd.concat(series_map.values(), axis=1)
    df_aligned.columns = names
    df_aligned_dropna = df_aligned.dropna()
    common_months = len(df_aligned_dropna)
    print(f"\n共同月份：{common_months} 个月（{df_aligned_dropna.index.min()} ~ {df_aligned_dropna.index.max()}）")

    # 相关性矩阵
    print_correlation_matrix(names, df_aligned_dropna)

    # 计算所有组合统计
    results: list[tuple[str, dict]] = []

    # 单策略（使用各自完整数据，不强制对齐）
    for name, series in series_map.items():
        results.append((name, portfolio_stats(series)))

    # 两两组合
    for a, b in combinations(names, 2):
        combo_series = build_portfolio([series_map[a], series_map[b]])
        combo_name = f"{a} + {b}"
        results.append((combo_name, portfolio_stats(combo_series)))

    # 三策略及以上组合
    if len(names) >= 3:
        for r in range(3, len(names) + 1):
            for combo in combinations(names, r):
                combo_series = build_portfolio([series_map[n] for n in combo])
                combo_name = " + ".join(combo)
                results.append((combo_name, portfolio_stats(combo_series)))

    print_stats_table(results)
    print_recommendation(results, len(names))

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
