"""
因子动量诊断（2026-08-14）：低成本自相关性检验，先于任何实现方案

背景：用户提出"行业层面因子动量轮动"作为新候选方向。但按最自然的方式实现
（用历史滚动因子收益/IC动态调整因子权重）与第六轮已证伪的"因子权重收缩
正则化"（James-Stein风格τ网格，`factor_shrinkage_v1.py`）本质上是同一类
自适应权重机制——第六轮结论：所有τ取值OOS均劣于固定基线，越自适应越差，
与ETF轮动"信号不可削弱"结论同源（详见 a_stock/docs/research.md 第六轮
小节）。

为避免直接搭建自适应权重系统重复验证已知失败模式，先做最低成本诊断：
检验"因子动量"这个前提本身在A股是否存在——即trailing因子收益是否能预测
下月因子收益（自相关性）。如果自相关性本身就弱或不稳定，说明连"因子动量"
的前提都不成立，不需要往下做任何实现方案，直接证伪，节省调研成本。

复用 factor_multi_backtest_v3_ablation.py 已预计算的月度因子数据
（reversal/ep_sector/ocf/roe/profit_stability/turnover/sue，中证500），
对每个因子计算月度多空收益（Top30% - Bottom30%），再检验该收益序列的
1个月自相关系数。

用法：
  cd a_stock/backtest
  python factor_momentum_diagnosis.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, DATA_DIR  # noqa: E402
from fetch_financials import load_financials  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_multi_backtest_v3_ablation import (  # noqa: E402
    get_industry_map, load_field_panel, load_circ_mv_panel,
    compute_turnover_daily, precompute_monthly_data, MEMBERS_FILE,
    FACTOR_NAMES,
)


def compute_long_short_return(entry: dict, fname: str) -> float | None:
    """单因子单月多空收益：Top30% 均值 - Bottom30% 均值（下月收益率）"""
    factors = entry["factors"]
    if fname not in factors:
        return None
    fs = factors[fname]
    fwd_ret = entry["fwd_ret"]
    common = fs.index.intersection(fwd_ret.index)
    if len(common) < 15:
        return None
    fs = fs[common]
    ret = fwd_ret[common]
    n_tier = max(int(len(fs) * 0.3), 3)
    top = fs.nlargest(n_tier).index
    bot = fs.nsmallest(n_tier).index
    return ret[top].mean() - ret[bot].mean()


def main():
    members = pd.read_parquet(MEMBERS_FILE)
    codes = members["con_code"].unique().tolist()

    print(f"加载收盘价面板（中证500，共 {len(codes)} 只股票）...")
    close_panel = load_close_panel(codes=codes)

    print("加载换手率所需 amount/circ_mv 面板...")
    amount_panel = load_field_panel(codes, "amount")
    circ_mv_panel = load_circ_mv_panel()
    turnover_panel = compute_turnover_daily(amount_panel, circ_mv_panel)

    print("加载申万行业映射...")
    industry_map = get_industry_map()

    print("预计算月度截面因子...")
    entries = precompute_monthly_data(close_panel, turnover_panel, industry_map)
    print(f"有效月份数：{len(entries)}\n")

    print("=" * 70)
    print("因子动量诊断：月度多空收益的1个月自相关系数")
    print("（本月多空收益 vs 上月多空收益，正相关=有动量可用，负相关=均值回归，"
          "接近0=无因子动量，不值得做自适应权重）")
    print("=" * 70)

    results = []
    for fname in FACTOR_NAMES:
        ls_rets = []
        dates = []
        for entry in entries:
            r = compute_long_short_return(entry, fname)
            ls_rets.append(r)
            dates.append(entry["month_end"])
        ser = pd.Series(ls_rets, index=dates).dropna()
        if len(ser) < 24:
            print(f"  {fname}: 样本不足（n={len(ser)}），跳过")
            continue

        lag1_autocorr = ser.autocorr(lag=1)

        # 分年度自相关，检验稳定性（防止全样本被单一区间主导）
        ser_df = pd.DataFrame({"ret": ser})
        ser_df["year"] = ser_df.index.year
        yearly_autocorr = {}
        for y in sorted(ser_df["year"].unique()):
            y_ser = ser_df[ser_df["year"] == y]["ret"]
            if len(y_ser) >= 6:
                yearly_autocorr[y] = y_ser.autocorr(lag=1)

        same_sign = np.mean([np.sign(v) == np.sign(lag1_autocorr) for v in yearly_autocorr.values()
                              if not np.isnan(v)]) if yearly_autocorr else np.nan

        results.append({
            "factor": fname,
            "n_months": len(ser),
            "月均多空收益": ser.mean(),
            "多空收益标准差": ser.std(),
            "1月自相关": lag1_autocorr,
            "年度同向占比": same_sign,
        })

        print(f"\n  {fname}:")
        print(f"    月均多空收益={ser.mean()*100:+.2f}%  标准差={ser.std()*100:.2f}%  "
              f"1月自相关={lag1_autocorr:+.3f}  年度同向占比={same_sign*100:.1f}%" if not np.isnan(same_sign) else
              f"    月均多空收益={ser.mean()*100:+.2f}%  标准差={ser.std()*100:.2f}%  1月自相关={lag1_autocorr:+.3f}")
        for y, v in yearly_autocorr.items():
            print(f"      {y}: {v:+.3f}")

    result_df = pd.DataFrame(results)
    out_dir = pathlib.Path(__file__).parent / "results" / "factor_momentum_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "autocorr_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("汇总（按|1月自相关|排序）：")
    print(result_df.sort_values("1月自相关", key=abs, ascending=False)[
        ["factor", "n_months", "1月自相关", "年度同向占比"]
    ].to_string(index=False))
    print(f"\n输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
