"""
日频截面 vs 月度截面 IC 对比实验（Qlib+RD-Agent 方向 POC，阶段1）

背景：第八轮外部调研诊断本项目多因子选股方向的核心瓶颈之一是样本量太小
（月度截面 10 年约 120 个观测点，机构常用日频截面一个月就上千个）。
用户提出"日频=交易磨损"的顾虑，本实验用于验证：日频截面采样（提升训练/检验
样本密度）与日频交易执行是两个独立维度——本实验只提高**因子计算和IC检验**的
采样频率，不涉及实际调仓频率，因此不产生任何额外交易成本。

方法：
- 复用 factor_ic_reversal.py 的反转因子定义（过去 window 日累计收益，反向使用）
- label 统一用未来 20 个交易日收益（贴近实际月度调仓的预测尺度，保证两种
  采样方式的 IC 可比）
- 对比两种采样方式：
    1. 月度采样：每月末算一次因子和 label（现状，约120个截面/10年）
    2. 日频采样：每个交易日算一次因子和 label（本实验新增，约2500个截面/10年）
- Point-in-time 成分股快照，与现有脚本口径一致

结论解读：
- 若日频采样 IC 显著高于月度采样（|ICIR| 提升、样本量增大后标准误下降），
  说明样本量瓶颈确实是月度截面损失信息导致的，Qlib+Alpha158 全量因子库值得投入
- 若日频/月度 IC 基本一致，说明瓶颈不在采样频率本身（可能是因子定义本身弱、
  或A股选股alpha结构性稀缺），日频改造收益有限

用法：
  cd a_stock/backtest
  python factor_ic_reversal_daily_vs_monthly.py --index hs500
  python factor_ic_reversal_daily_vs_monthly.py --index hs300
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR

INDEX_CONFIG = {
    "hs300": {"name": "沪深300", "members_file": DATA_DIR / "hs300_members.parquet", "min_stocks": 50},
    "hs500": {"name": "中证500", "members_file": DATA_DIR / "hs500_members.parquet", "min_stocks": 80},
}

START_DATE = "2016-01-01"
END_DATE = "2026-06-30"
REVERSAL_WINDOW = 63   # 沿用现有结论里最强的窗口（中证500 63日反转 ICIR -0.123）
LABEL_HORIZON = 20     # 未来20个交易日收益，两种采样方式统一用这个尺度
MIN_STOCKS_PER_CROSS = 50


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


def compute_ic_series(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    dates: np.ndarray,
) -> pd.DataFrame:
    """
    在给定的采样日期序列上（月末或逐日），计算 window 日反转因子 vs 未来
    LABEL_HORIZON 日收益的截面 Rank IC。
    """
    idx = close_panel.index
    records = []
    for i, d in enumerate(dates):
        pos = idx.searchsorted(d)
        if pos >= len(idx) or idx[pos] != d:
            continue
        if pos < REVERSAL_WINDOW or pos + LABEL_HORIZON >= len(idx):
            continue

        pit_members = load_members_pit(d, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        factor = close_panel[available].iloc[pos] / close_panel[available].iloc[pos - REVERSAL_WINDOW] - 1
        factor = winsorize(factor.dropna())
        factor = standardize(factor)

        now_price = close_panel[available].iloc[pos]
        fwd_price = close_panel[available].iloc[pos + LABEL_HORIZON]
        fwd_ret = (fwd_price / now_price - 1).dropna()

        common = factor.index.intersection(fwd_ret.index)
        ic = cross_section_rank_ic(factor[common], fwd_ret[common])
        records.append({"date": d, "ic": ic, "n_stocks": len(common)})

    return pd.DataFrame(records).set_index("date")


def summarize(ic_df: pd.DataFrame, label: str) -> None:
    ic = ic_df["ic"].dropna()
    icir = ic.mean() / ic.std() if ic.std() > 1e-8 else np.nan
    pos_pct = (ic > 0).mean()
    # 简单的 IC 均值显著性检验（假设近似独立，实际截面间有重叠会低估标准误，仅供粗略参考）
    se = ic.std() / np.sqrt(len(ic))
    t_stat = ic.mean() / se if se > 1e-8 else np.nan
    print(f"\n[{label}]")
    print(f"  截面数: {len(ic)}")
    print(f"  IC均值: {ic.mean():.4f}")
    print(f"  ICIR:   {icir:.4f}")
    print(f"  t统计量（粗略，未校正自相关): {t_stat:.2f}")
    print(f"  IC>0占比: {pos_pct:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=["hs300", "hs500"], default="hs500")
    args = parser.parse_args()

    cfg = INDEX_CONFIG[args.index]
    print(f"=== {cfg['name']}：{REVERSAL_WINDOW}日反转因子，label=未来{LABEL_HORIZON}日收益 ===")

    close_panel = load_close_panel().loc[START_DATE:END_DATE]

    # 月度采样：月末交易日
    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_dates = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    # 日频采样：全部交易日（重叠窗口，样本间有自相关，仅用于对比信息密度，不代表可独立交易的信号数）
    daily_dates = close_panel.index.values

    ic_monthly = compute_ic_series(close_panel, cfg["members_file"], monthly_dates)
    ic_daily = compute_ic_series(close_panel, cfg["members_file"], daily_dates)

    summarize(ic_monthly, "月度采样（现状口径）")
    summarize(ic_daily, "日频采样（逐日滚动）")

    print("\n注：日频采样的截面之间因子/label高度重叠（相邻交易日几乎是同一组股票同一价格窗口），")
    print("t统计量和ICIR会被自相关严重高估，不能直接当作'样本量增加N倍、显著性增加sqrt(N)倍'来看。")
    print("这里只用 IC 均值本身（不受自相关影响）判断日频信息是否比月度采样点携带更多信号，")
    print("如果两者 IC 均值接近，说明瓶颈不在采样频率；如果日频明显更强，才值得往 Qlib 方向投入。")


if __name__ == "__main__":
    main()
