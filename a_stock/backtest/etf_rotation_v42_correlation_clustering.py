"""
相关性聚类候选池分散约束实测（机械化候选池版）

背景：v28行业cap测试证明"每行业砍到1-2只"的静态分层约束在431只候选池上
反而恶化夏普（越整齐候选池、可选空间越窄，机械化动量策略越吃亏，与v34同
指数去重失败模式一致）。相关性聚类理论上可以避开这个失败模式：聚类分组
基于实际收益相关性而非静态行业标签，且贪心法允许同cluster内保留多只
（cap>1时不是"分组内只留1个代表"的强约束）。用户2026-07-27决策"跳过"，
本轮（2026-08-13）用户明确要求补测这个此前唯一未被数据证伪的方向。

聚类方法：全样本日收益率相关性矩阵 → 距离=1-corr → 层次聚类（average
linkage）→ 按距离阈值切割（阈值取1-CORR_THRESHOLD=0.30，对应
etf_rotation.py现有CORR_THRESHOLD=0.70口径，不引入新的任意参数）。
静态聚类（一次性，非滚动重聚类）：与现有use_industry_cap机制完全对齐
（industry_map本身就是静态字典），先验证最简单版本是否有效，再考虑是否
需要动态重聚类。

复用 etf_rotation.py 现有 use_industry_cap 贪心法框架：把静态申万
industry_map 换成聚类映射 cluster_map（ts_code -> cluster_id），机制本身
不变，不修改 etf_rotation.py。

方法论：全部在431只机械化候选池上验证，不用45只手工池（用户2026-07-27
决策）。
"""

import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_stats, CASH_ETF, CORR_THRESHOLD  # noqa: E402
from etf_rotation import run_backtest  # noqa: E402
from etf_rotation_v27_concentration_risk import build_universe_close_and_scores  # noqa: E402

MIN_OVERLAP_DAYS = 60  # 与 etf_rotation.py CORR_WINDOW 一致，重叠不足视为不相关


def build_cluster_map(close: pd.DataFrame, corr_threshold: float) -> dict:
    """全样本日收益率相关性 → 层次聚类（average linkage）→ 按距离阈值切割。"""
    rets = close.pct_change().dropna(how="all")
    codes = list(close.columns)
    n = len(codes)

    corr = rets.corr(min_periods=MIN_OVERLAP_DAYS)
    dist = 1 - corr.values
    dist = np.nan_to_num(dist, nan=1.0)  # 重叠不足：视为不相关（距离=1）
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # 消除浮点不对称
    dist[dist < 0] = 0.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")

    return {codes[i]: int(cluster_ids[i]) for i in range(n)}


def main():
    print("重建机械化候选池、价格矩阵、动量得分...")
    close, masked_scores, rebal_dates = build_universe_close_and_scores()
    print(f"候选池标的数：{close.shape[1]}，调仓日数量：{len(rebal_dates)}")

    print("\n计算全样本收益相关性矩阵，层次聚类...")
    cluster_map_70 = build_cluster_map(close, corr_threshold=CORR_THRESHOLD)
    sizes_70 = pd.Series(cluster_map_70.values()).value_counts()
    print(f"阈值corr>0.70：{len(sizes_70)}个cluster，"
          f"最大cluster{sizes_70.max()}只，单标的cluster占比"
          f"{(sizes_70 == 1).sum()}/{len(sizes_70)}")

    cluster_map_60 = build_cluster_map(close, corr_threshold=0.60)
    sizes_60 = pd.Series(cluster_map_60.values()).value_counts()
    print(f"阈值corr>0.60（敏感性对照）：{len(sizes_60)}个cluster，"
          f"最大cluster{sizes_60.max()}只，单标的cluster占比"
          f"{(sizes_60 == 1).sum()}/{len(sizes_60)}")

    configs = [
        ("基线（无约束）", dict(use_industry_cap=False)),
        ("聚类cap=1（corr>0.70）", dict(use_industry_cap=True, industry_map=cluster_map_70, max_per_industry=1)),
        ("聚类cap=2（corr>0.70）", dict(use_industry_cap=True, industry_map=cluster_map_70, max_per_industry=2)),
        ("聚类cap=1（corr>0.60，敏感性）", dict(use_industry_cap=True, industry_map=cluster_map_60, max_per_industry=1)),
    ]

    print("\n" + "=" * 70)
    print("相关性聚类分散约束对比（机械化候选池，431只）")
    print("=" * 70)
    rows = []
    for label, kwargs in configs:
        nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF, **kwargs)
        stats = calc_stats(nav, label)
        rows.append(stats)
        print(f"\n{label}:")
        for k, v in stats.items():
            if k != "标的":
                print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(pd.DataFrame(rows).set_index("标的").to_string())


if __name__ == "__main__":
    main()
