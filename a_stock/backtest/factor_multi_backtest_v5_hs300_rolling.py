"""
沪深300最优子集滚动窗口稳健性检验（2026-09-20）

背景：`factor_multi_backtest_v5_hs300_resmom.py`的127子集消融显示
`ep_sector+roe+resmom_252s21`OOS超额年化+6.2%，跑赢V2基线（-5.0%）。但
127种子集里挑OOS表现最大值本身构成多重检验（trading-standards.md的
n_trials规则），全样本/IS/OOS三段划分样本量有限，不足以排除运气成分。

本脚本按`trading-standards.md`"滚动窗口稳健性检验"标准，对比两个子集：
- 候选：ep_sector + roe + resmom_252s21
- 基线：V2（reversal + ep_sector + ocf + roe + profit_stability）

滚动36个月窗口（3年），逐窗口比较候选超额 vs 基线超额，统计候选劣于基线
的窗口占比。判定标准：劣于基线占比>30%视为不稳健，18.8-18.9%视为稳健，
20-25%为灰色地带。

用法：
  cd a_stock/backtest
  python factor_multi_backtest_v5_hs300_rolling.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_multi_backtest_v2 import (  # noqa: E402
    get_industry_map,
    COST_PER_TRADE, STAMP_DUTY,
)
from factor_multi_backtest_v5_hs300_resmom import (  # noqa: E402
    precompute_monthly_data, run_subset_backtest, MEMBERS_FILE, INDEX_NAME,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_multi_backtest_v5_hs300_rolling"

ROLLING_WINDOW = 36  # 3年，与trading-standards.md滚动窗口稳健性检验一致

CANDIDATE_SUBSET = ("ep_sector", "roe", "resmom_252s21")
BASELINE_SUBSET  = ("reversal", "ep_sector", "ocf", "roe", "profit_stability")


def rolling_excess(ret_df: pd.DataFrame) -> pd.Series:
    """逐窗口滚动超额收益年化"""
    excess = ret_df["strategy"] - ret_df["benchmark"]
    return excess.rolling(ROLLING_WINDOW).mean().dropna() * 12


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    members = pd.read_parquet(MEMBERS_FILE)
    codes = members["con_code"].unique().tolist()

    print(f"加载收盘价面板（{INDEX_NAME}，共 {len(codes)} 只股票）...")
    close_panel = load_close_panel(codes=codes)

    print("加载申万行业映射...")
    industry_map = get_industry_map()

    print("预计算月度截面因子...")
    entries = precompute_monthly_data(close_panel, industry_map)
    print(f"有效月份数：{len(entries)}\n")

    print(f"候选子集：{'+'.join(CANDIDATE_SUBSET)}")
    print(f"基线子集：{'+'.join(BASELINE_SUBSET)}\n")

    cand_ret_df = run_subset_backtest(entries, CANDIDATE_SUBSET)
    base_ret_df = run_subset_backtest(entries, BASELINE_SUBSET)

    common_dates = cand_ret_df.index.intersection(base_ret_df.index)
    cand_ret_df = cand_ret_df.loc[common_dates].sort_index()
    base_ret_df = base_ret_df.loc[common_dates].sort_index()

    cand_excess_roll = rolling_excess(cand_ret_df)
    base_excess_roll = rolling_excess(base_ret_df)

    common_roll = cand_excess_roll.index.intersection(base_excess_roll.index)
    cand_roll = cand_excess_roll.loc[common_roll]
    base_roll = base_excess_roll.loc[common_roll]

    n_windows = len(common_roll)
    worse_mask = cand_roll < base_roll
    n_worse = worse_mask.sum()
    worse_pct = n_worse / n_windows * 100 if n_windows > 0 else np.nan

    print("=" * 60)
    print(f"滚动{ROLLING_WINDOW}个月窗口稳健性检验")
    print("=" * 60)
    print(f"滚动窗口数量：{n_windows}")
    print(f"候选子集滚动超额年化：均值={cand_roll.mean()*100:+.2f}%  标准差={cand_roll.std()*100:.2f}%")
    print(f"基线子集滚动超额年化：均值={base_roll.mean()*100:+.2f}%  标准差={base_roll.std()*100:.2f}%")
    print(f"候选劣于基线的窗口占比：{n_worse}/{n_windows} = {worse_pct:.1f}%")

    if worse_pct > 30:
        verdict = "不稳健（>30%）"
    elif worse_pct <= 20:
        verdict = "稳健（<=20%，参照历史判例18.8-18.9%）"
    else:
        verdict = "灰色地带（20-30%，需结合其他证据）"
    print(f"判定：{verdict}")

    detail = pd.DataFrame({
        "候选滚动超额年化": cand_roll,
        "基线滚动超额年化": base_roll,
        "候选劣于基线": worse_mask.loc[common_roll],
    })
    detail.to_csv(OUTPUT_DIR / "rolling_comparison.csv")

    print(f"\n候选子集滚动超额年化为负的窗口占比：{(cand_roll < 0).sum()}/{n_windows} = {(cand_roll < 0).mean()*100:.1f}%")
    print(f"基线子集滚动超额年化为负的窗口占比：{(base_roll < 0).sum()}/{n_windows} = {(base_roll < 0).mean()*100:.1f}%")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
