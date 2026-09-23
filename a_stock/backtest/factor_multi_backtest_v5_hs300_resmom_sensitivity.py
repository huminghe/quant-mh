"""
第四十六轮候选（ep_sector+roe+resmom_252s21）接入实盘前的三项验证补测（2026-09-23）

背景：`factor_multi_backtest_v5_hs300_resmom.py`已完成IC初筛+组合消融+滚动36个月
稳健性检验，但`research_index_enhancement.md`第1062行标注还有三项缺口未补：
①resmom_126s21/252s21二选一敏感性（两者截面相关性+0.686，只验证过252日窗口）
②不同IS/OOS切分点下OOS超额是否稳健（排除2024-02这一个切分点的偶然性）
③n_trials=127的DSR/Bonferroni多重检验校正（trading-standards.md要求n_trials>10时
  必须做，此前只做了"排序取最大值"）

复用v5的月度因子预计算（`precompute_monthly_data`），避免重新拉数据。

用法：
  cd a_stock/backtest
  python factor_multi_backtest_v5_hs300_resmom_sensitivity.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, DATA_DIR  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_multi_backtest_v2 import get_industry_map, sharpe, max_drawdown  # noqa: E402
from factor_multi_backtest_v5_hs300_resmom import (  # noqa: E402
    precompute_monthly_data, run_subset_backtest, MEMBERS_FILE, INDEX_NAME,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_multi_backtest_v5_hs300_resmom_sensitivity"

BEST_SUBSET_252 = ("ep_sector", "roe", "resmom_252s21")
BEST_SUBSET_126 = ("ep_sector", "roe", "resmom_126s21")
V2_BASELINE = ("reversal", "ep_sector", "ocf", "roe", "profit_stability")

# 三个IS/OOS切分点，跨度覆盖2023-08~2024-08，排除单一切分点偶然性
SPLIT_POINTS = [
    ("2023-08-31", "2023-09-01"),
    ("2024-01-31", "2024-02-01"),  # 原始切分点，与v5一致
    ("2024-08-31", "2024-09-01"),
]

N_TRIALS_ABLATION = 127  # v5全子集消融的候选总数，DSR校正口径


def deflated_sharpe_ratio_fixed(returns: pd.Series, num_trials: int, annualization: float = 12 ** 0.5) -> dict:
    """
    修正版DSR，复用`etf_rotation_v33_pool_cap_pbo.py`已验证的本地实现（plugin的
    overfit_detector.deflated_sharpe_ratio存在expected_max_sr漏乘sr_std的量纲bug，
    不直接调用）。annualization默认按月度数据传入sqrt(12)，与本脚本月度收益序列匹配。
    """
    observed_sr = returns.mean() / returns.std() * annualization
    sr = observed_sr / annualization
    n = len(returns)
    skew = float(returns.skew())
    kurt = float(returns.kurt() + 3.0)

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


def evaluate_with_split(ret_df: pd.DataFrame, is_end: str, oos_start: str) -> dict:
    if ret_df.empty:
        return None
    strat = ret_df["strategy"]
    bench = ret_df["benchmark"]
    excess = strat - bench

    is_mask = ret_df.index <= is_end
    oos_mask = ret_df.index >= oos_start
    nav_s = (1 + strat).cumprod()

    return {
        "n_months": len(ret_df),
        "IS超额年化": excess[is_mask].mean() * 12 if is_mask.sum() > 0 else np.nan,
        "OOS超额年化": excess[oos_mask].mean() * 12 if oos_mask.sum() > 0 else np.nan,
        "OOS月份数": int(oos_mask.sum()),
        "全样本夏普": sharpe(strat),
        "全样本最大回撤": max_drawdown(nav_s),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    members = pd.read_parquet(MEMBERS_FILE)
    codes = members["con_code"].unique().tolist()

    print(f"加载收盘价面板（{INDEX_NAME}，共 {len(codes)} 只股票）...")
    close_panel = load_close_panel(codes=codes)
    print("加载申万行业映射...")
    industry_map = get_industry_map()

    print("预计算月度截面因子（复用v5逻辑）...")
    entries = precompute_monthly_data(close_panel, industry_map)
    print(f"有效月份数：{len(entries)}\n")

    # ── 验证①：resmom_126s21 vs resmom_252s21 二选一敏感性 ──
    print("=" * 60)
    print("验证①：resmom窗口二选一敏感性（252日 vs 126日）")
    print("=" * 60)
    rows_1 = []
    for label, subset in [("ep_sector+roe+resmom_252s21", BEST_SUBSET_252),
                           ("ep_sector+roe+resmom_126s21", BEST_SUBSET_126)]:
        ret_df = run_subset_backtest(entries, subset)
        stats = evaluate_with_split(ret_df, "2024-01-31", "2024-02-01")
        stats["subset"] = label
        rows_1.append(stats)
    df1 = pd.DataFrame(rows_1)
    print(df1[["subset", "IS超额年化", "OOS超额年化", "全样本夏普", "全样本最大回撤"]]
          .to_string(index=False, formatters={
              "IS超额年化": lambda x: f"{x*100:+.1f}%",
              "OOS超额年化": lambda x: f"{x*100:+.1f}%",
              "全样本夏普": lambda x: f"{x:.3f}",
              "全样本最大回撤": lambda x: f"{x*100:.1f}%",
          }))
    df1.to_csv(OUTPUT_DIR / "resmom_window_sensitivity.csv", index=False)

    # ── 验证②：不同IS/OOS切分点敏感性 ──
    print(f"\n{'='*60}")
    print("验证②：不同切分点下 ep_sector+roe+resmom_252s21 的OOS表现")
    print(f"{'='*60}")
    ret_df_best = run_subset_backtest(entries, BEST_SUBSET_252)
    ret_df_v2 = run_subset_backtest(entries, V2_BASELINE)
    rows_2 = []
    for is_end, oos_start in SPLIT_POINTS:
        s_best = evaluate_with_split(ret_df_best, is_end, oos_start)
        s_v2 = evaluate_with_split(ret_df_v2, is_end, oos_start)
        rows_2.append({
            "切分点": f"{is_end}/{oos_start}",
            "候选OOS超额年化": s_best["OOS超额年化"],
            "候选OOS月份数": s_best["OOS月份数"],
            "V2基线OOS超额年化": s_v2["OOS超额年化"],
            "候选跑赢基线": s_best["OOS超额年化"] > s_v2["OOS超额年化"],
        })
    df2 = pd.DataFrame(rows_2)
    print(df2.to_string(index=False, formatters={
        "候选OOS超额年化": lambda x: f"{x*100:+.1f}%",
        "V2基线OOS超额年化": lambda x: f"{x*100:+.1f}%",
    }))
    df2.to_csv(OUTPUT_DIR / "split_point_sensitivity.csv", index=False)

    # ── 验证③：n_trials=127 DSR校正 ──
    print(f"\n{'='*60}")
    print(f"验证③：DSR多重检验校正（n_trials={N_TRIALS_ABLATION}）")
    print(f"{'='*60}")
    dsr_best = deflated_sharpe_ratio_fixed(ret_df_best["strategy"], num_trials=N_TRIALS_ABLATION)
    dsr_v2 = deflated_sharpe_ratio_fixed(ret_df_v2["strategy"], num_trials=1)  # 基线不是挑出来的，不做多重检验校正
    print(f"候选子集 ep_sector+roe+resmom_252s21：")
    print(f"  观测夏普（年化） = {dsr_best['observed_sr']:.3f}")
    print(f"  期望最大夏普（零真实alpha下，{N_TRIALS_ABLATION}候选） = {dsr_best['expected_max_sr']:.3f}")
    print(f"  DSR p值 = {dsr_best['dsr_pvalue']:.3f}（显著性阈值0.95）")
    print(f"  是否通过DSR校正后仍显著 = {dsr_best['is_significant']}")
    print(f"\nV2基线（未参与本轮挑选，不做多重检验校正，仅作对比）：")
    print(f"  观测夏普（年化） = {dsr_v2['observed_sr']:.3f}")

    pd.DataFrame([
        {"子集": "ep_sector+roe+resmom_252s21", **dsr_best, "n_trials": N_TRIALS_ABLATION},
    ]).to_csv(OUTPUT_DIR / "dsr_correction.csv", index=False)

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
