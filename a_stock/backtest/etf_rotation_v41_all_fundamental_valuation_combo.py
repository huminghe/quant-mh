"""
第十四轮·问题②收尾：7个未通过IC初筛的基本面/估值指标，逐一叠加到动量策略实测（2026-07-27）

背景：v37/v39/v40对7个财务/估值指标做IC检验，仅ocf_to_profit通过初筛（且
v38组合消融证明它在组合层面无增量贡献）。其余6个（netprofit_yoy/roe_delta/
debt_to_assets/or_yoy/pe_ttm/pb）因IC检验未过筛，此前从未进入组合回测。

用户追问"可以测一下和动量一起用的效果如何"——参考第十二轮flow信号的先例
（个体IC未达标，但集成后组合层面转正），IC筛选只是成本较低的初筛，不能
100%排除"组合层面隐藏价值"的可能，值得补一次直接验证，避免用间接推断
代替实测（第十三轮已有moneyflow_ratio/rate_beta的反例：IC未过筛的信号
在63子集全量消融里也确实从未进入最优子集，说明大多数情况IC初筛是可靠的，
但仍需实测确认，不能凭先例跳过）。

方法：对7个信号（含已知ocf_to_profit，作为对照）逐一做"连续打折叠加到
动量基线"回测，同时测一次7信号等权集成版本。信号方向（invert）按v37/v39/v40
实测的全样本IC均值符号决定：IC为正不invert，IC为负invert。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import get_rebalance_dates, MOMENTUM_WINDOW, TOP_N, START_DATE  # noqa: E402
from etf_rotation_v23_universe_bias_test import load_close_matrix_from_cache  # noqa: E402
from etf_rotation_v37_industry_fundamental_ic import (  # noqa: E402
    EXPOSURE_FILE, get_sw_industry_map, get_etf_industry_map,
    build_industry_fundamental_panel, industry_signal_to_etf,
)
from etf_rotation_v40_valuation_ic import build_industry_valuation_panel  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import calc_risk_adj_momentum  # noqa: E402
from etf_rotation_v18_signal_ablation import run_backtest, calc_stats, roll_sharpe  # noqa: E402

IS_RATIO = 0.8

# (信号key, 取值字段, 是否用delta, 全样本IC符号是否invert, 展示名)
SIGNAL_SPECS = [
    ("netprofit_yoy", "netprofit_yoy", False, True,  "净利润同比增速(IC-0.0036,几乎为零仍按符号处理)"),
    ("roe_delta",      "roe_dt",        True,  False, "ROE环比变化(IC+0.0161)"),
    ("ocf_to_profit",  "ocf_to_profit", False, False, "现金流质量OCF/NI(IC+0.0373，对照，已知v38组合无贡献)"),
    ("debt_to_assets", "debt_to_assets", False, False, "资产负债率/杠杆(IC+0.0113)"),
    ("or_yoy",         "or_yoy",        False, True,  "营业收入同比增速(IC-0.0079)"),
]
VALUATION_SPECS = [
    ("pe_ttm", "pe_ttm边界IC+0.0184，信号已取负，不再invert"),
    ("pb",     "pb IC+0.0017，信号已取负，不再invert"),
]


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

    etf_industry_map = get_etf_industry_map()
    sector_codes = [c for c in valid_codes if c in etf_industry_map]
    industry_map = get_sw_industry_map()
    all_stock_codes = list(industry_map.keys())
    print(f"有申万行业映射的行业ETF：{len(sector_codes)} 只，全A股财务个股池：{len(all_stock_codes)} 只")

    print("\n构建7个信号面板...")
    etf_signals = {}
    for key, field, use_delta, invert, name in SIGNAL_SPECS:
        panel = build_industry_fundamental_panel(all_stock_codes, industry_map, field, use_delta, rebal_dates)
        etf_sig = industry_signal_to_etf(panel, etf_industry_map, sector_codes)
        etf_sig = etf_sig[[c for c in etf_sig.columns if c in valid_codes]]
        etf_signals[key] = (etf_sig, invert)
        print(f"  {name} 完成")

    for key, name in VALUATION_SPECS:
        panel = build_industry_valuation_panel(key, industry_map)
        etf_sig = industry_signal_to_etf(panel, etf_industry_map, sector_codes)
        etf_sig = etf_sig[[c for c in etf_sig.columns if c in valid_codes]]
        etf_signals[key] = (etf_sig, False)  # v40已取负，不再invert
        print(f"  {name} 完成")

    print("\n构建各信号的月度rank...")
    signal_ranks = {name: {} for name in etf_signals}
    for d in rebal_dates:
        for name, (sig_df, invert) in etf_signals.items():
            if sig_df.empty:
                continue
            idx = sig_df.index[sig_df.index <= d]
            if len(idx) == 0:
                continue
            s_d = sig_df.loc[idx[-1]].dropna()
            if len(s_d) < 5:
                continue
            r = s_d.rank(pct=True)
            if invert:
                r = 1 - r
            signal_ranks[name][d] = r

    rank_dfs = {name: pd.DataFrame(d).T for name, d in signal_ranks.items()}

    def combo_rank(names):
        parts = [rank_dfs[n] for n in names if not rank_dfs[n].empty]
        if not parts:
            return pd.DataFrame()
        common_dates = parts[0].index
        for other in parts[1:]:
            common_dates = common_dates.union(other.index)
        combined = {}
        for d in common_dates:
            vals = [p.loc[d] for p in parts if d in p.index]
            if vals:
                combined[d] = pd.concat(vals, axis=1).mean(axis=1)
        return pd.DataFrame(combined).T

    n_days = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]
    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]
    print(f"\nIS/OOS拆分点：{split_date.date()}（IS {len(rebal_is)}个调仓月，OOS {len(rebal_oos)}个调仓月）")

    print("\n" + "=" * 90)
    print("7个基本面/估值指标逐一叠加到动量策略实测（连续打折叠加，与v38同方法）")
    print("=" * 90)

    rows = []
    nav_cache = {}
    nav_base = run_backtest(close, scores, rebal_dates, top_n=TOP_N)
    stats_base = calc_stats(nav_base)
    nav_base_is = run_backtest(close_is, sc_is, rebal_is, top_n=TOP_N)
    nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos, top_n=TOP_N)
    rows.append({"信号": "（无，纯动量基线）", "夏普": stats_base["Sharpe"], "年化": stats_base["CAGR"],
                 "回撤": stats_base["MaxDD"], "IS夏普": calc_stats(nav_base_is)["Sharpe"],
                 "OOS夏普": calc_stats(nav_base_oos)["Sharpe"]})
    nav_cache["（无，纯动量基线）"] = nav_base

    all_keys = list(etf_signals.keys())
    for key in all_keys:
        boost = combo_rank([key])
        nav = run_backtest(close, scores, rebal_dates, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        stats = calc_stats(nav)
        nav_is = run_backtest(close_is, sc_is, rebal_is, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
        rows.append({"信号": key, "夏普": stats["Sharpe"], "年化": stats["CAGR"],
                     "回撤": stats["MaxDD"], "IS夏普": calc_stats(nav_is)["Sharpe"],
                     "OOS夏普": calc_stats(nav_oos)["Sharpe"]})
        nav_cache[key] = nav
        print(f"  完成: {key}  夏普={stats['Sharpe']:.3f}")

    # 7信号等权集成
    boost_all = combo_rank(all_keys)
    nav_ens = run_backtest(close, scores, rebal_dates, boost_signal=boost_all, boost_mode="continuous", top_n=TOP_N)
    stats_ens = calc_stats(nav_ens)
    nav_ens_is = run_backtest(close_is, sc_is, rebal_is, boost_signal=boost_all, boost_mode="continuous", top_n=TOP_N)
    nav_ens_oos = run_backtest(close_oos, sc_oos, rebal_oos, boost_signal=boost_all, boost_mode="continuous", top_n=TOP_N)
    rows.append({"信号": "7信号等权集成", "夏普": stats_ens["Sharpe"], "年化": stats_ens["CAGR"],
                 "回撤": stats_ens["MaxDD"], "IS夏普": calc_stats(nav_ens_is)["Sharpe"],
                 "OOS夏普": calc_stats(nav_ens_oos)["Sharpe"]})
    nav_cache["7信号等权集成"] = nav_ens
    print(f"  完成: 7信号等权集成  夏普={stats_ens['Sharpe']:.3f}")

    df = pd.DataFrame(rows).set_index("信号")
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
    print(f"结论：最优信号「{best_label}」夏普={best_sharpe:.3f}，基线={baseline_sharpe:.3f}，Δ={delta:+.3f}")
    print("=" * 90)

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
    print("全部信号夏普排序：")
    print(others.sort_values("夏普", ascending=False)[["夏普", "年化", "回撤"]].to_string())
    print("=" * 90)


if __name__ == "__main__":
    main()
