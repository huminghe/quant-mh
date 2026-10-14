"""
第十四轮换源候选A：行业残差动量IC初筛（2026-10-14）

背景：第十四轮换源调研（海外机构研报/学术新论文方向）发现SSRN论文《The
Anatomy of Industry Momentum》（Gao/Li/Yuan/Zhou, 2026-03/09）——该文把
行业动量信号分解为"系统成分"和"残差成分"，发现alpha主要来自把短期反转
污染剥离后的残差成分，剥离后的信号比原始行业动量更强、更稳定。

本脚本把这个机制搬到ETF行业轮动场景做IC初筛：
  1. 用现有calc_risk_adj_momentum()算标准风险调整动量（25日OLS斜率×R²÷
     21日年化波动率，项目主信号，已上线）作为"原始动量"基线。
  2. 构造短期反转信号：21日收益率（符号不翻转，原始值）。
  3. 对每个月度评估截面，用当期全部ETF的"原始动量 ~ 短期反转"做横截面OLS
     回归，取残差——这就是"剥离了短期反转污染后的残差动量"，是SSRN论文
     机制在截面上的直接对应（时间序列上的"系统成分/残差成分"分解，在
     截面场景下对应于"按反转回归后的残差"）。
  4. 用v37同款IC检验框架（evaluate_signal/report_ic，|IC均值|>=0.03且
     年度同向占比>=60%通过初筛）测残差动量信号的月度截面Rank IC。
  5. 用v37同款冗余检验（cross_section_corr，阈值0.5）与现有信号（原始
     动量/crowding/vol_ratio/flow）比较，排除高相关性的冗余信号。

与v17原始动量/crowding/vol_ratio的区别：那些是已上线的现成信号本身；
本脚本构造的是"原始动量的变体"（剔除反转污染后的残差），如果IC通过且
与原始动量截面相关性够低，才有替换或补充原始动量的价值；如果残差后
IC不升反降，或者与原始动量高度相关，则说明A股ETF层面动量信号本身
反转污染不严重，候选A的机制假设不成立。

覆盖范围：与v37一致，用etf_sw_exposure.parquet的industry类（排除HK:
前缀）筛选行业ETF子集做横截面回归和IC检验——残差动量的构造依赖"同一
截面内有足够多标的做OLS回归"，行业ETF子集（约170+只，覆盖24个申万
一级行业）比全部431只候选池更适合（宽基/QDII没有"行业"属性，混进去
会稀释回归关系，而非把异质性当噪声吸收）。

判定标准（复用v17/v37既定阈值）：
  - |IC均值| >= 0.03 且年度同向占比 >= 60%：通过初筛
  - 与现有信号（动量/crowding/vol_ratio/flow）截面相关性 > 0.5：视为冗余
  - 不达标：直接排除，不进入组合层面消融（v38方法论）
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import get_rebalance_dates, MOMENTUM_WINDOW, START_DATE  # noqa: E402
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    load_close_matrix_from_cache,
)
from etf_rotation_v25_universe_ensemble_backtest import load_amount_matrix_from_cache  # noqa: E402
from etf_rotation_v37_industry_fundamental_ic import (  # noqa: E402
    get_etf_industry_map, evaluate_signal, report_ic, cross_section_corr,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import (  # noqa: E402
    calc_risk_adj_momentum, calc_crowding, fetch_fund_share_all,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_FILE = DATA_DIR / "etf_all_candidates.parquet"

REVERSAL_WINDOW = 21  # 短期反转窗口（约1个月交易日），与风险调整动量的波动率窗口一致


def calc_short_term_reversal(close: pd.DataFrame, window: int = REVERSAL_WINDOW) -> pd.DataFrame:
    """短期反转信号：过去window日累计收益率（原始值，不翻转符号）。
    回归时让OLS自己估计系数方向，不预设"反转"的符号假设，更贴近SSRN
    论文"剥离短期反转污染"而非"叠加反转因子"的机制定位。"""
    return close.pct_change(window)


def cross_section_residual_momentum(mom: pd.DataFrame, reversal: pd.DataFrame) -> pd.DataFrame:
    """
    逐个评估日做横截面OLS回归：原始动量 ~ 短期反转，取残差。
    残差即"剥离了短期反转污染后的动量"——如果某ETF的高动量很大程度上
    能被其短期反转水平解释（例如动量得分高主要是因为最近正好经历了
    一轮反转修复），回归会把这部分共性剔除，只留下回归无法解释的部分。
    """
    common_dates = mom.index.intersection(reversal.index)
    residual = pd.DataFrame(index=common_dates, columns=mom.columns, dtype=float)
    for d in common_dates:
        m = mom.loc[d].dropna()
        r = reversal.loc[d].dropna()
        common_codes = m.index.intersection(r.index)
        if len(common_codes) < 10:  # 截面回归至少需要足够样本量才有意义
            continue
        m_vals = m[common_codes].values
        r_vals = r[common_codes].values
        if np.std(r_vals) < 1e-10:
            continue
        slope, intercept = np.polyfit(r_vals, m_vals, 1)
        fitted = slope * r_vals + intercept
        resid = m_vals - fitted
        residual.loc[d, common_codes] = resid
    return residual


def main():
    print("加载ETF候选池与价格缓存...")
    all_candidates = pd.read_parquet(EXPOSURE_FILE)["ts_code"].tolist()
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    etf_industry_map = get_etf_industry_map()
    sector_codes = [c for c in valid_codes if c in etf_industry_map]
    print(f"其中有申万行业映射（etf_sw_exposure.parquet）的行业ETF：{len(sector_codes)} 只，"
          f"覆盖 {len(set(etf_industry_map[c] for c in sector_codes))} 个申万一级行业")

    fwd_1m = close.pct_change().rolling(21).sum().shift(-21)
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("\n计算风险调整动量（原始基线信号）...")
    mom_scores = calc_risk_adj_momentum(close_full)[valid_codes]
    mom_scores = mom_scores[mom_scores.index >= START_DATE]

    print("计算短期反转信号（21日累计收益率）...")
    reversal_sig = calc_short_term_reversal(close_full[valid_codes])
    reversal_sig = reversal_sig[reversal_sig.index >= START_DATE]

    print("\n在行业ETF子集上做横截面OLS回归，构造残差动量信号...")
    mom_sector = mom_scores[sector_codes]
    reversal_sector = reversal_sig[sector_codes]
    residual_mom = cross_section_residual_momentum(mom_sector, reversal_sector)
    n_valid_dates = residual_mom.notna().any(axis=1).sum()
    print(f"  残差动量面板：{n_valid_dates} 个有效日期，{residual_mom.notna().any().sum()} 只标的至少一期非空")

    print("\n计算现有候选信号（crowding/vol_ratio/flow），用于冗余检验对照...")
    crowding = calc_crowding(close_full[valid_codes])
    crowding = crowding[crowding.index >= START_DATE]
    amount = load_amount_matrix_from_cache(valid_codes)
    amount = amount[amount.index >= START_DATE]
    vol_ratio_sig = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_sig = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    print("\n" + "=" * 90)
    print("诊断1：残差动量 vs 原始动量 —— 单独IC对比（月度截面Rank IC，行业ETF子集）")
    print("=" * 90)

    results = {}
    for key, name, sig in [
        ("residual_momentum", "残差动量（剔除短期反转污染）", residual_mom),
        ("raw_momentum_sector", "原始风险调整动量（行业ETF子集，对照）", mom_sector),
    ]:
        ic = evaluate_signal(sig, fwd_1m, rebal_dates)
        passed, ic_mean = report_ic(name, ic)
        results[key] = {"name": name, "signal": sig, "ic": ic, "passed": passed, "ic_mean": ic_mean}

    print("\n" + "=" * 90)
    print("诊断2：与主信号（全池动量）+ 现有候选信号的截面相关性检验（冗余判定，阈值0.5）")
    print("=" * 90)
    reference_signals = {
        "动量(全池)": mom_scores, "crowding": crowding, "vol_ratio": vol_ratio_sig, "flow": flow_sig,
    }
    for key, r in results.items():
        name = r["name"]
        sig = r["signal"]
        max_abs_corr = 0.0
        max_ref_name = ""
        for ref_name, ref_sig in reference_signals.items():
            if ref_sig.empty:
                continue
            corr_mean = cross_section_corr(sig, ref_sig)
            if pd.isna(corr_mean):
                continue
            print(f"  {name:<30} vs {ref_name:<12}  相关性均值={corr_mean:+.4f}")
            if abs(corr_mean) > abs(max_abs_corr):
                max_abs_corr = corr_mean
                max_ref_name = ref_name
        redundant = abs(max_abs_corr) > 0.5
        print(f"  {name:<30}  最大相关性来自「{max_ref_name}」={max_abs_corr:+.4f}  "
              f"{'冗余（排除）' if redundant else '独立'}")
        results[key]["redundant"] = redundant

    print("\n" + "=" * 90)
    print("最终判定：")
    print("=" * 90)
    survivors = []
    for key, r in results.items():
        final_pass = r["passed"] and not r["redundant"]
        print(f"  {r['name']:<30}  IC达标={r['passed']}  冗余={r['redundant']}  "
              f"→ {'进入组合消融（参照v38）' if final_pass else '排除'}")
        if final_pass:
            survivors.append(key)

    print("\n--- 残差动量 vs 原始动量 IC对比结论 ---")
    res_ic = results["residual_momentum"]["ic_mean"]
    raw_ic = results["raw_momentum_sector"]["ic_mean"]
    if abs(res_ic) > abs(raw_ic):
        print(f"残差动量IC均值({res_ic:+.4f})强于原始动量({raw_ic:+.4f})，"
              f"剔除短期反转污染后信号增强，SSRN机制在ETF行业轮动场景下成立。")
    else:
        print(f"残差动量IC均值({res_ic:+.4f})未强于原始动量({raw_ic:+.4f})，"
              f"剔除短期反转污染没有带来信号增强，SSRN机制在本场景下不成立"
              f"（可能原因：A股ETF层面动量信号本身反转污染不严重，或25日窗口"
              f"动量与21日窗口反转重叠度过高导致回归把真实动量成分也一并剔除）。")

    if survivors:
        print(f"\n通过初筛的信号：{survivors}，可考虑做组合层面消融（参照v38方法论）。")
    else:
        print("\n候选A（行业残差动量）未通过初筛，判定排除，不进入组合层面消融。")


if __name__ == "__main__":
    main()
