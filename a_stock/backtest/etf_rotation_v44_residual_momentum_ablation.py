"""
第十四轮候选A：残差动量信号并入候选池后的组合层面消融（2026-10-14）

背景：v43对候选A（行业残差动量，SSRN《The Anatomy of Industry Momentum》
机制）做IC初筛——IC均值+0.0307（刚过0.03阈值，边际通过），年度同向占比
70.0%，与动量/crowding/vol_ratio/flow截面相关性均<0.4（独立）。但IC只是
初筛，边际通过的信号必须进一步做组合层面消融才能判断是否有真实增量，
避免IC初筛阶段的运气成分被误判为有效信号。

候选池现扩大为5个弱信号：crowding、vol_ratio、flow、ocf_to_profit（v37/
v38已验证通过）、residual_momentum（v43新验证通过，本脚本加入）。
对全部31种非空子集做等权排名集成消融，统一用v16/v18/v38验证最优的
"连续打折"叠加方式，找出含残差动量后是否仍是最优子集，并对最优子集做
滚动2年窗口稳健性检验（方法论完全沿用v38，仅追加第5个信号）。

残差动量信号覆盖范围：与v43一致，限定在etf_sw_exposure.parquet的
industry类ETF子集（197只，24个申万一级行业），非行业ETF（宽基/QDII）
不参与该信号的boost，combo_rank对多信号取均值时按天动态跳过缺失部分
（与flow/vol_ratio/ocf_to_profit的覆盖处理方式一致）。

基准候选池：机械化431只候选池（`etf_all_candidates.parquet`）。
"""

import sys
import pathlib
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import get_rebalance_dates, MOMENTUM_WINDOW, TOP_N  # noqa: E402
from etf_rotation_v23_universe_bias_test import load_close_matrix_from_cache  # noqa: E402
from etf_rotation_v25_universe_ensemble_backtest import load_amount_matrix_from_cache  # noqa: E402
from etf_rotation_v37_industry_fundamental_ic import (  # noqa: E402
    get_sw_industry_map, get_etf_industry_map, build_industry_fundamental_panel,
    industry_signal_to_etf, FUNDAMENTAL_FIELDS,
)
from etf_rotation_v43_residual_momentum_ic import (  # noqa: E402
    calc_short_term_reversal, cross_section_residual_momentum,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import (  # noqa: E402
    calc_risk_adj_momentum, calc_crowding, fetch_fund_share_all,
)
from etf_rotation_v18_signal_ablation import run_backtest, calc_stats, roll_sharpe  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_FILE = DATA_DIR / "etf_all_candidates.parquet"

START_DATE = "2016-01-01"
IS_RATIO = 0.8


def main():
    print("加载机械化候选池与价格缓存...")
    all_candidates = pd.read_parquet(EXPOSURE_FILE)["ts_code"].tolist()
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    print("计算动量得分（基线）...")
    scores = calc_risk_adj_momentum(close_full)[valid_codes]
    scores = scores[scores.index >= START_DATE]
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("计算拥挤度信号...")
    crowding = calc_crowding(close_full[valid_codes])
    crowding = crowding[crowding.index >= START_DATE]

    print("计算成交量确认信号...")
    amount = load_amount_matrix_from_cache(valid_codes)
    amount = amount[amount.index >= START_DATE]
    vol_ratio = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取ETF份额数据（资金流信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    print("构建现金流质量(OCF/NI)行业景气度信号（v37已验证通过初筛）...")
    etf_industry_map = get_etf_industry_map()
    sector_codes = [c for c in valid_codes if c in etf_industry_map]
    print(f"  有申万行业映射的行业ETF：{len(sector_codes)} 只")
    industry_map = get_sw_industry_map()
    all_stock_codes = list(industry_map.keys())
    ocf_panel = build_industry_fundamental_panel(
        all_stock_codes, industry_map,
        FUNDAMENTAL_FIELDS["ocf_to_profit"]["field"],
        FUNDAMENTAL_FIELDS["ocf_to_profit"]["delta"],
        rebal_dates,
    )
    ocf_etf = industry_signal_to_etf(ocf_panel, etf_industry_map, sector_codes)
    ocf_etf = ocf_etf[[c for c in ocf_etf.columns if c in valid_codes]]

    print("构建残差动量信号（v43已验证通过初筛，IC+0.0307边际通过）...")
    reversal_sig = calc_short_term_reversal(close_full[valid_codes])
    reversal_sig = reversal_sig[reversal_sig.index >= START_DATE]
    mom_sector = scores[sector_codes]
    reversal_sector = reversal_sig[sector_codes]
    residual_mom = cross_section_residual_momentum(mom_sector, reversal_sector)
    residual_mom = residual_mom[[c for c in residual_mom.columns if c in valid_codes]]

    all_names = ["crowding", "vol_ratio", "flow", "ocf_to_profit", "residual_momentum"]
    signal_ranks = {name: {} for name in all_names}
    for d in rebal_dates:
        crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
        volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)
        flow_d = pd.Series(dtype=float)
        if not flow_1m.empty:
            idx = flow_1m.index[flow_1m.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m.loc[idx[-1]]
        ocf_d = pd.Series(dtype=float)
        if not ocf_etf.empty:
            idx = ocf_etf.index[ocf_etf.index <= d]
            if len(idx) > 0:
                ocf_d = ocf_etf.loc[idx[-1]]
        resid_d = pd.Series(dtype=float)
        if not residual_mom.empty:
            idx = residual_mom.index[residual_mom.index <= d]
            if len(idx) > 0:
                resid_d = residual_mom.loc[idx[-1]]

        for name, s, invert in [("crowding", crowd_d, True), ("vol_ratio", volr_d, False),
                                 ("flow", flow_d, True), ("ocf_to_profit", ocf_d, False),
                                 ("residual_momentum", resid_d, False)]:
            s = s.dropna()
            if len(s) < 5:
                continue
            r = s.rank(pct=True)
            if invert:
                r = 1 - r
            signal_ranks[name][d] = r

    rank_dfs = {name: pd.DataFrame(d).T for name, d in signal_ranks.items()}

    def combo_rank(names):
        parts = [rank_dfs[n] for n in names if not rank_dfs[n].empty]
        if not parts:
            return pd.DataFrame()
        combined = {}
        common_dates = parts[0].index
        for other in parts[1:]:
            common_dates = common_dates.union(other.index)
        for d in common_dates:
            vals = [p.loc[d] for p in parts if d in p.index]
            if vals:
                combined[d] = pd.concat(vals, axis=1).mean(axis=1)
        return pd.DataFrame(combined).T

    subsets = []
    for r in range(1, len(all_names) + 1):
        subsets.extend(itertools.combinations(all_names, r))
    print(f"\n全部非空子集数：{len(subsets)}")

    n_days = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]
    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]
    print(f"IS/OOS拆分点：{split_date.date()}（IS {len(rebal_is)}个调仓月，OOS {len(rebal_oos)}个调仓月）")

    print("\n" + "=" * 90)
    print("信号子集消融实验：动量基线 + 5个候选弱信号的全部非空子集（连续打折叠加）")
    print("=" * 90)

    rows = []
    nav_cache = {}

    nav_base = run_backtest(close, scores, rebal_dates, top_n=TOP_N)
    stats_base = calc_stats(nav_base)
    nav_base_is = run_backtest(close_is, sc_is, rebal_is, top_n=TOP_N)
    nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos, top_n=TOP_N)
    rows.append({"信号子集": "（无，纯动量基线）", "夏普": stats_base["Sharpe"], "年化": stats_base["CAGR"],
                 "回撤": stats_base["MaxDD"], "IS夏普": calc_stats(nav_base_is)["Sharpe"],
                 "OOS夏普": calc_stats(nav_base_oos)["Sharpe"]})
    nav_cache["（无，纯动量基线）"] = nav_base

    for names in subsets:
        label = "+".join(names)
        boost = combo_rank(list(names))
        nav = run_backtest(close, scores, rebal_dates, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        stats = calc_stats(nav)
        nav_is = run_backtest(close_is, sc_is, rebal_is, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        rows.append({"信号子集": label, "夏普": stats["Sharpe"], "年化": stats["CAGR"],
                     "回撤": stats["MaxDD"], "IS夏普": calc_stats(nav_is)["Sharpe"],
                     "OOS夏普": calc_stats(nav_oos)["Sharpe"]})
        nav_cache[label] = nav
        print(f"  完成: {label}  夏普={stats['Sharpe']:.3f}")

    df = pd.DataFrame(rows).set_index("信号子集")
    df_fmt = df.copy()
    df_fmt["夏普"] = df_fmt["夏普"].map(lambda x: f"{x:.3f}")
    df_fmt["年化"] = df_fmt["年化"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["回撤"] = df_fmt["回撤"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["IS夏普"] = df_fmt["IS夏普"].map(lambda x: f"{x:.3f}")
    df_fmt["OOS夏普"] = df_fmt["OOS夏普"].map(lambda x: f"{x:.3f}")
    print("\n" + df_fmt.to_string())

    baseline_sharpe = df.loc["（无，纯动量基线）", "夏普"]
    others = df.drop("（无，纯动量基线）")
    best_label = others["夏普"].idxmax()
    best_sharpe = others.loc[best_label, "夏普"]
    delta = best_sharpe - baseline_sharpe

    print("\n" + "=" * 90)
    print(f"结论：最优信号子集「{best_label}」夏普={best_sharpe:.3f}，基线={baseline_sharpe:.3f}，Δ={delta:+.3f}")
    print("=" * 90)

    # 对照：含residual_momentum的最优子集 vs 不含的最优子集（判断该信号是否真正贡献增量）
    without_resid = others[~others.index.str.contains("residual_momentum")]
    with_resid = others[others.index.str.contains("residual_momentum")]
    if not without_resid.empty and not with_resid.empty:
        best_wo_label = without_resid["夏普"].idxmax()
        best_wo_sharpe = without_resid.loc[best_wo_label, "夏普"]
        best_with_label = with_resid["夏普"].idxmax()
        best_with_sharpe = with_resid.loc[best_with_label, "夏普"]
        print(f"对照：不含residual_momentum的最优子集「{best_wo_label}」夏普={best_wo_sharpe:.3f}")
        print(f"对照：含residual_momentum的最优子集「{best_with_label}」夏普={best_with_sharpe:.3f}")
        print(f"residual_momentum信号的边际贡献Δ={best_with_sharpe - best_wo_sharpe:+.3f}")

    if delta > 0.02:
        nav_best = nav_cache[best_label]
        window_days = 252 * 2
        rolling_base, rolling_best = [], []
        for i in range(window_days, len(nav_base)):
            rolling_base.append((nav_base.index[i], roll_sharpe(nav_base.iloc[i - window_days: i])))
        for i in range(window_days, len(nav_best)):
            rolling_best.append((nav_best.index[i], roll_sharpe(nav_best.iloc[i - window_days: i])))
        rs_base = pd.Series(dict(rolling_base))
        rs_best = pd.Series(dict(rolling_best))
        common_idx = rs_base.index.intersection(rs_best.index)
        improvement = rs_best[common_idx] - rs_base[common_idx]
        neg_ratio = (improvement < 0).mean()
        print(f"\n滚动2年夏普稳健性检验（「{best_label}」 vs 基线）：")
        print(f"均值Δ={improvement.mean():+.3f}，std={improvement.std():.3f}，劣于基线占比={neg_ratio:.1%}")
        if neg_ratio > 0.4 or improvement.std() > abs(improvement.mean()) * 2:
            print("→ 不稳健，判定过拟合痕迹，不建议采用。")
        else:
            print("→ 相对稳健，可考虑小规模试用并持续监控。")
    else:
        print(f"\n提升Δ={delta:+.3f} <= 0.02阈值，不做滚动窗口稳健性检验。")

    print("\n" + "=" * 90)
    print("全部子集夏普排序（前15）：")
    print(others.sort_values("夏普", ascending=False)[["夏普", "年化", "回撤"]].head(15).to_string())
    print("=" * 90)


if __name__ == "__main__":
    main()
