"""
ETF轮动策略 PBO 计算（第十轮调研 Task 3）
读取 etf_rotation_v10_pbo_collect.py 生成的收益矩阵，调用
overfit_detector.py 的 probability_of_backtest_overfitting 计算PBO。

注：该函数经审查未发现与 deflated_sharpe_ratio 同类的量纲bug
（全程用相对排名+logit变换，不涉及SR标准误换算），可直接使用。

**方法修正说明（重要）**：最初把59条净值序列（覆盖10个历史脚本、多个互不
相关的独立决策——调仓频率/含QDII/协方差收缩/波动率目标/拥挤度过滤）
一起塞进CPCV，得到PBO=0.533。这个用法不成立：CPCV要求候选来自同一个
搜索空间、回答同一个调参决策，混合多个决策后"IS-best"在15条路径里
换成完全不同性质的候选，复合结果没有清晰的现实含义。
下方改为只对候选数足够多（≥7）的单一决策网格分别计算PBO，
即波动率目标网格（13候选）和拥挤度过滤网格（7候选）。其余网格
（协方差收缩3候选、调仓频率3候选等）候选数太少，CPCV统计意义不足，
不纳入。
"""

import pathlib
import sys

import pandas as pd

SKILL_DIR = pathlib.Path.home() / ".claude/plugins/marketplaces/agiprolabs-claude-trading-skills/skills/walk-forward-validation/scripts"
sys.path.insert(0, str(SKILL_DIR))
from overfit_detector import probability_of_backtest_overfitting

CSV_PATH = pathlib.Path(__file__).parent.parent / "results" / "v10_pbo_returns_matrix.csv"

ret_df = pd.read_csv(CSV_PATH, index_col=0)
print(f"收益矩阵：{ret_df.shape[0]} 个观测 × {ret_df.shape[1]} 个策略")

# ── 已废弃：59候选混合矩阵的PBO（方法不成立，仅保留供追溯） ──────────
result = probability_of_backtest_overfitting(ret_df.values, n_groups=6, n_test_groups=2)
print(f"\n[已废弃，方法不成立] 59候选混合PBO = {result.pbo:.3f}"
      f"（{result.n_overfit_paths}/{result.n_paths}路径过拟合）")

# IS最优策略是哪个（供报告引用）
# 注：etf_rotation_analysis.py 的 nav_full 是该脚本内部网格搜索出的全样本
# 最优参数组合（非当前上线配置），不能作为基线标签；当前上线基线（Top3/25日/
# 风险调整动量+QDII）在多个脚本中都以相同数值出现，用 v9_qdii_ic::nav_full
# 作为代表。
BASELINE_LABEL = "etf_rotation_v9_qdii_ic::nav_full"
is_best_idx = ret_df.mean().values.argmax()
print(f"全样本均值最高的策略：{ret_df.columns[is_best_idx]}")
print(f"当前上线基线（{BASELINE_LABEL}）在全样本均值排名："
      f"{(ret_df.mean() > ret_df.mean()[BASELINE_LABEL]).sum() + 1} / {ret_df.shape[1]}")

# ── 修正：对单一决策网格分别计算PBO ──────────────────────────────
print(f"\n{'=' * 60}\n修正：单一决策网格分别计算PBO")
for prefix, name in [
    ("etf_rotation_v9_voltarget", "波动率目标网格（13候选，target_vol×lookback）"),
    ("etf_rotation_v3b_fullval", "拥挤度过滤网格（7候选，阈值组合）"),
]:
    cols = [c for c in ret_df.columns if c.startswith(prefix)]
    sub = ret_df[cols]
    r = probability_of_backtest_overfitting(sub.values, n_groups=6, n_test_groups=2)
    print(f"\n{name}")
    print(f"  PBO = {r.pbo:.3f}（{r.n_overfit_paths}/{r.n_paths}路径过拟合）"
          f"  平均OOS排名={r.mean_oos_rank:.3f}  is_overfit={r.is_overfit}")
