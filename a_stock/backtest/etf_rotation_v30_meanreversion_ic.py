"""
非动量选股方法 B-1：均值回归信号IC检验（机械化候选池，2026-07-27）

背景：ETF轮动60+轮测试全部是"动量框架内叠加/过滤"，从未测试过跟动量原理相反的
独立选股逻辑。均值回归假设：短期超跌的行业ETF更可能反弹，短期超涨的更可能回落，
与当前上线的"追涨"动量逻辑方向相反。

本脚本只做IC检验（不进组合回测），在机械化431只候选池上：
  信号 = -过去N日累计收益率（N分别测20/60日），值越大代表越"超跌"
  前瞻收益 = 未来21个交易日累计收益率
  月度截面Rank IC，判定标准复用项目既定阈值：
    |IC均值| >= 0.03 且年度同向占比 >= 60% 视为通过初筛

同时报告与当前风险调整动量信号的截面相关性，判断均值回归是否只是动量的镜像
（预期强负相关，这本身不是排除理由，只是确认逻辑方向相反）。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_all_scores, get_rebalance_dates, MOMENTUM_WINDOW, START_DATE  # noqa: E402
from etf_rotation_v27_concentration_risk import build_universe_close_and_scores  # noqa: E402

REVERSION_WINDOWS = [20, 60]
FWD_WINDOW = 21  # 未来21个交易日（约1个月），与调仓频率一致


def calc_reversion_signal(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """信号 = -过去window日累计收益率，值越大代表越超跌"""
    past_ret = close.pct_change(window)
    return -past_ret


def cross_section_rank_ic(factor: pd.Series, fwd: pd.Series) -> float:
    common = factor.dropna().index.intersection(fwd.dropna().index)
    if len(common) < 5:
        return np.nan
    return factor[common].corr(fwd[common], method="spearman")


def evaluate_signal(signal: pd.DataFrame, fwd: pd.DataFrame, rebal_dates: list) -> pd.Series:
    ic_list = []
    for d in rebal_dates:
        if d not in fwd.index or d not in signal.index:
            continue
        s_d = signal.loc[d].dropna()
        f_d = fwd.loc[d]
        ic = cross_section_rank_ic(s_d, f_d)
        if not pd.isna(ic):
            ic_list.append((d, ic))
    return pd.Series(dict(ic_list))


def report_ic(name: str, ic: pd.Series) -> bool:
    if ic.empty:
        print(f"  {name}: 无有效样本")
        return False
    yearly = ic.groupby(ic.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(ic.mean())).mean() if ic.mean() != 0 else 0
    passed = abs(ic.mean()) >= 0.03 and same_sign >= 0.6
    print(f"  {name:<20}  IC均值={ic.mean():+.4f}  IC>0占比={(ic > 0).mean():.1%}  "
          f"年度同向占比={same_sign:.1%}  样本={len(ic)}月  "
          f"{'通过初筛' if passed else '未达阈值'}")
    print(f"    年度拆解：{dict(yearly.round(4))}")
    return passed


def cross_section_corr(sig_a: pd.DataFrame, sig_b: pd.DataFrame, dates: list) -> float:
    corrs = []
    for d in dates:
        if d not in sig_a.index or d not in sig_b.index:
            continue
        a = sig_a.loc[d].dropna()
        b = sig_b.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        corrs.append(a[common].corr(b[common], method="spearman"))
    return float(np.nanmean(corrs)) if corrs else np.nan


def main():
    print("重建机械化候选池、价格矩阵...")
    close, mom_scores, rebal_dates = build_universe_close_and_scores()
    print(f"候选池标的数：{close.shape[1]}，调仓日数量：{len(rebal_dates)}")

    fwd = close.pct_change(FWD_WINDOW).shift(-FWD_WINDOW)

    print("\n" + "=" * 90)
    print("均值回归信号IC检验（不同回看窗口）")
    print("=" * 90)

    results = {}
    for window in REVERSION_WINDOWS:
        name = f"均值回归(回看{window}日)"
        signal = calc_reversion_signal(close, window)
        ic = evaluate_signal(signal, fwd, rebal_dates)
        passed = report_ic(name, ic)
        results[name] = {"signal": signal, "ic": ic, "passed": passed}

    print("\n" + "=" * 90)
    print("与当前风险调整动量信号的截面相关性（预期强负相关，确认逻辑方向相反）")
    print("=" * 90)
    for name, r in results.items():
        corr = cross_section_corr(r["signal"], mom_scores, rebal_dates)
        print(f"  {name:<20} vs 风险调整动量  相关性均值={corr:+.4f}")

    print("\n" + "=" * 90)
    print("最终判定：")
    print("=" * 90)
    survivors = [name for name, r in results.items() if r["passed"]]
    if survivors:
        print(f"通过初筛：{survivors}，值得进一步做组合回测（替换/混合当前动量评分函数）。")
    else:
        print("均值回归各回看窗口均未通过IC初筛，不建议进入组合回测。")


if __name__ == "__main__":
    main()
