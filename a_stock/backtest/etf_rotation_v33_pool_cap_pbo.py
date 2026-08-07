"""
候选池门槛×行业分散约束 联合决策网格的PBO/DSR稳健性检验（2026-07-27）

背景：v31确认0.1亿/0.2亿门槛候选池比当前1.0亿门槛更抗冲击成本；v32发现在扩容
候选池上cap=2优于cap=1（与431只池上v28的结论方向相反）。这些都是"矩阵表格里
挑最大值"的操作，用户要求做PBO/DSR检验这个联合决策本身是否过拟合。

决策空间：门槛∈{1.0, 0.1, 0.2}亿 × 行业约束∈{基线, cap=1, cap=2}，共9个候选，
这是同一个调参决策（候选池升级方向的联合选择），不与其他历史决策（动量窗口、
Top N等）混合，满足trading-standards.md的PBO候选集合要求。

方法：
1. 对9个候选分别跑回测，取日收益序列，对齐公共日期区间，构建(n_obs, 9)矩阵。
2. 用overfit_detector.probability_of_backtest_overfitting做CPCV，n_groups=6,
   n_test_groups=2（与项目历史用法一致，见v10_pbo_result.py/v12_riskadj_grid_pbo.py）。
3. 对全样本夏普最高的候选，用deflated_sharpe_ratio计算DSR，num_trials=9
   （本决策网格候选数，不与其他历史轮次的试验数混用口径——按trading-standards.md
   "同一份历史数据上比较过的候选总数"计，这是一次独立的候选池+cap联合决策，
   不是对"当前上线策略核心参数"的检验，num_trials不应累加v10/v12等历史试验数）。

注意：1.0亿门槛的cap用旧`etf_sw_exposure.parquet`（431只，v26/v27/v28同源）；
0.1亿/0.2亿门槛的cap用新`etf_sw_exposure_01yi.parquet`（1143只，v32同源）。
两个文件覆盖范围不同但对应的候选池本身也不同，不存在混用问题。
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_stats, run_backtest, CASH_ETF  # noqa: E402
from etf_rotation_v23_universe_bias_test import TURNOVER_PATH, META_PATH  # noqa: E402
from etf_rotation_v31_impact_cost_sensitivity import build_pool  # noqa: E402

SKILL_DIR = pathlib.Path.home() / ".claude/plugins/marketplaces/agiprolabs-claude-trading-skills/skills/walk-forward-validation/scripts"
sys.path.insert(0, str(SKILL_DIR))
from overfit_detector import probability_of_backtest_overfitting  # noqa: E402


def deflated_sharpe_ratio_fixed(returns: pd.Series, num_trials: int, annualization: float = 252 ** 0.5) -> dict:
    """
    修正版DSR（第十轮已发现原版plugin的量纲bug：expected_max_sr忘记乘sr_std就直接
    annualize，导致"期望最大夏普"离谱地大于任何真实观测夏普，DSR恒等于0）。
    公式来自 Bailey & Lopez de Prado (2014)，本地重新实现，不依赖plugin的
    deflated_sharpe_ratio 函数。
    """
    import numpy as np
    from scipy.stats import norm

    observed_sr = returns.mean() / returns.std() * annualization
    sr = observed_sr / annualization  # de-annualize
    n = len(returns)
    skew = float(returns.skew())
    kurt = float(returns.kurt() + 3.0)  # pandas .kurt() 是超额峰度，+3还原为正常峰度

    sr_std = np.sqrt((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1))

    euler_mascheroni = 0.5772156649
    if num_trials <= 1:
        expected_max_sr = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / num_trials)
        z2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        expected_max_sr = sr_std * (z1 * (1.0 - euler_mascheroni) + euler_mascheroni * z2)

    p = norm.cdf((sr - expected_max_sr) / sr_std) if sr_std > 0 else (1.0 if sr > expected_max_sr else 0.0)

    return {
        "observed_sr": observed_sr,
        "expected_max_sr": expected_max_sr * annualization,
        "dsr_pvalue": p,
        "is_significant": p > 0.95,
    }

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_PATH_OLD = DATA_DIR / "etf_sw_exposure.parquet"        # 431只（1.0亿门槛）
EXPOSURE_PATH_NEW = DATA_DIR / "etf_sw_exposure_01yi.parquet"   # 1143只（0.1/0.2亿门槛）
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

THRESHOLDS_YI = [1.0, 0.1, 0.2]


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]

    industry_map_old = pd.read_parquet(EXPOSURE_PATH_OLD).set_index("ts_code")["dominant_industry"].to_dict()
    industry_map_new = pd.read_parquet(EXPOSURE_PATH_NEW).set_index("ts_code")["dominant_industry"].to_dict()

    navs = {}
    rows = []
    for th in THRESHOLDS_YI:
        print(f"\n{'=' * 70}\n构建门槛 {th} 亿元候选池...\n{'=' * 70}")
        close, masked_scores, rebal_dates, _amount_wide, pool_size = build_pool(turnover, th)
        print(f"候选池规模：{pool_size}只，调仓日数量：{len(rebal_dates)}")

        industry_map = industry_map_old if th == 1.0 else industry_map_new
        configs = [
            ("基线（无约束）", dict(use_industry_cap=False)),
            ("行业cap=1", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=1)),
            ("行业cap=2", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=2)),
        ]

        for label, kwargs in configs:
            nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF, **kwargs)
            full_label = f"门槛{th}亿(候选{pool_size}只)_{label}"
            navs[full_label] = nav
            stats = calc_stats(nav, full_label)
            stats["门槛(亿元)"] = th
            stats["候选池规模"] = pool_size
            stats["配置"] = label
            rows.append(stats)
            print(f"  {label}: 夏普={stats['年化夏普']}, 回撤={stats['最大回撤']}")

    df = pd.DataFrame(rows).set_index("标的")
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "v33_pool_cap_grid.csv")

    print("\n" + "=" * 70)
    print("候选池×cap 9候选网格汇总")
    print("=" * 70)
    print(df.to_string())

    # ── PBO：9候选同一决策网格 ──────────────────────────────
    print(f"\n{'=' * 70}\nPBO 检验：9候选（门槛×cap联合决策，同一次搜索）\n{'=' * 70}")

    common_index = None
    for nav in navs.values():
        idx = nav.dropna().index
        common_index = idx if common_index is None else common_index.intersection(idx)
    print(f"公共日期区间：{common_index.min().date()} ~ {common_index.max().date()}，{len(common_index)}个交易日")

    rets = {label: nav.reindex(common_index).ffill().pct_change() for label, nav in navs.items()}
    ret_df = pd.DataFrame(rets).iloc[1:].dropna(axis=1)
    dropped = len(navs) - ret_df.shape[1]
    if dropped:
        print(f"[注意] {dropped} 条序列因对齐后仍含缺失值被丢弃")
    print(f"收益矩阵：{ret_df.shape[0]} 个观测 × {ret_df.shape[1]} 个候选")

    ret_df.to_csv(RESULTS_DIR / "v33_pool_cap_returns_matrix.csv")

    pbo_result = probability_of_backtest_overfitting(ret_df.values, n_groups=6, n_test_groups=2)
    print(f"\nPBO = {pbo_result.pbo:.3f}（{pbo_result.n_overfit_paths}/{pbo_result.n_paths}路径过拟合）")
    print(f"平均OOS排名 = {pbo_result.mean_oos_rank:.3f}  is_overfit = {pbo_result.is_overfit}")

    is_best_label = ret_df.mean().idxmax()
    print(f"\n全样本均值最高的候选：{is_best_label}")
    baseline_label = [c for c in ret_df.columns if c.startswith("门槛1.0亿") and "基线" in c][0]
    print(f"当前生产配置（{baseline_label}）在全样本均值排名："
          f"{(ret_df.mean() > ret_df.mean()[baseline_label]).sum() + 1} / {ret_df.shape[1]}")

    # ── DSR：对全样本夏普最高的候选 + 当前生产基线对照 ──────────
    print(f"\n{'=' * 70}\nDSR 检验（本地修正版公式，num_trials=9，本决策网格候选数）\n{'=' * 70}")
    n_trials = len(ret_df.columns)
    for label in [is_best_label, baseline_label]:
        r = ret_df[label]
        d = deflated_sharpe_ratio_fixed(r, num_trials=n_trials)
        tag = "全样本最优" if label == is_best_label else "当前生产基线"
        print(f"\n[{tag}] {label}")
        print(f"  观测年化夏普 = {d['observed_sr']:.3f}")
        print(f"  期望最大夏普（零真实alpha下，{n_trials}候选） = {d['expected_max_sr']:.3f}")
        print(f"  DSR p值 = {d['dsr_pvalue']:.3f}  is_significant(>0.95) = {d['is_significant']}")

    print(f"\n结果已保存至 {RESULTS_DIR / 'v33_pool_cap_grid.csv'}")


if __name__ == "__main__":
    main()
