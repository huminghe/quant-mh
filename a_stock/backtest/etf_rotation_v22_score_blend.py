"""
调仓日选股得分平滑：用"本月1日"与"上月25日"两个快照的得分加权平均，
代替单一日期的得分快照，再选Top3。执行调仓的日期仍是当月首个交易日不变。

背景：v21锚定日敏感性测试发现，OLS风险调整动量得分对窗口边界日期极度
敏感（边界点杠杆是窗口中点的144倍），且同月内相隔20个交易日的得分排名
相关性已跌到接近0，导致"哪天算分"这个选择被放大成"选出完全不同的一批
标的"（历史56.5%的月份，锚定日1和25选出的Top3持仓完全不重叠）。

本脚本测试一种缓解思路：用两个时间上接近（约一周内）的快照做加权平均，
而不是依赖单一日期的单点估计，观察能否降低这种窗口噪音、提升稳健性。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch_data import load_close_matrix
from etf_rotation_v16_signal_combo_ablation import calc_risk_adj_momentum
from etf_rotation_v21_fixed_calendar_days import (
    get_rebalance_dates_anchor_day, run_backtest, calc_stats,
)

MOMENTUM_WINDOW = 25
TOP_N = 3
IS_RATIO = 0.8
WEIGHTS = [1.0, 0.7, 0.5, 0.3, 0.0]  # 1.0=纯本月1日（基线），0.0=纯上月25日


def build_blended_scores(index: pd.DatetimeIndex, scores: pd.DataFrame, weight_now: float) -> dict:
    """对每个月的锚定日=1调仓点，构造 weight_now*本月1日得分 + (1-weight_now)*上月25日得分"""
    d1_list = get_rebalance_dates_anchor_day(index, 1)
    d25_list = get_rebalance_dates_anchor_day(index, 25)

    blended = {}
    for d1 in d1_list:
        if d1 not in scores.index:
            continue
        s_now = scores.loc[d1]
        prior_25 = [d for d in d25_list if d < d1]
        if not prior_25 or prior_25[-1] not in scores.index:
            # 首月缺上月25日数据时，退化为纯当月得分，保证调仓点集合与基线完全对齐
            blended[d1] = s_now
            continue
        d25 = prior_25[-1]
        s_prev = scores.loc[d25]
        common = s_now.index.union(s_prev.index)
        s_now_f = s_now.reindex(common)
        s_prev_f = s_prev.reindex(common)
        blend = weight_now * s_now_f.fillna(s_prev_f) + (1 - weight_now) * s_prev_f.fillna(s_now_f)
        blended[d1] = blend
    return blended


def report(label: str, close: pd.DataFrame, blended_scores: dict, split_date: pd.Timestamp) -> dict:
    rebal_dates = sorted(blended_scores.keys())
    scores_df = pd.DataFrame(blended_scores).T

    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is = scores_df[scores_df.index < split_date]
    sc_oos = scores_df[scores_df.index >= split_date]

    nav, meta = run_backtest(close, scores_df, rebal_dates)
    nav_is, _ = run_backtest(close_is, sc_is, rebal_is)
    nav_oos, _ = run_backtest(close_oos, sc_oos, rebal_oos)

    stats = calc_stats(nav)
    stats_is = calc_stats(nav_is)
    stats_oos = calc_stats(nav_oos)
    decay = stats_oos["Sharpe"] / stats_is["Sharpe"] if stats_is["Sharpe"] > 0 else 0

    return {"label": label, "nav": nav, "stats": stats, "stats_is": stats_is,
            "stats_oos": stats_oos, "decay": decay, "meta": meta, "rebal_dates": rebal_dates}


def turnover_vs_baseline(blended_scores: dict, baseline_scores: dict) -> float:
    """计算blended方案相对基线（纯1日）方案，Top3持仓的平均Jaccard重合度"""
    jaccards = []
    for d in sorted(set(blended_scores) & set(baseline_scores)):
        s1 = blended_scores[d].dropna()
        s2 = baseline_scores[d].dropna()
        top1 = set(s1[s1 > 0].nlargest(TOP_N).index)
        top2 = set(s2[s2 > 0].nlargest(TOP_N).index)
        if top1 or top2:
            jac = len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 1.0
            jaccards.append(jac)
    return np.mean(jaccards) if jaccards else np.nan


def main():
    print("加载价格数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= "2016-01-01"]
    valid = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid]
    print(f"有效标的：{len(valid)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

    scores = calc_risk_adj_momentum(close_full)[valid]
    scores = scores[scores.index >= "2016-01-01"]

    split_idx = int(len(close) * IS_RATIO)
    split_date = close.index[split_idx]

    print("\n" + "=" * 95)
    print("本月1日得分 与 上月25日得分 加权平均，Top3选股，调仓仍在本月1日执行")
    print("=" * 95)

    results = []
    baseline_scores = None
    for w in WEIGHTS:
        blended = build_blended_scores(close.index, scores, w)
        if w == 1.0:
            baseline_scores = blended
        label = f"w(本月1日)={w:.1f}"
        r = report(label, close, blended, split_date)
        r["blended_scores"] = blended
        results.append(r)
        print(f"  {label:<16}  夏普={r['stats']['Sharpe']:.3f}  年化={r['stats']['CAGR']*100:.1f}%  "
              f"回撤={r['stats']['MaxDD']*100:.1f}%  IS={r['stats_is']['Sharpe']:.3f}  "
              f"OOS={r['stats_oos']['Sharpe']:.3f}  调仓次数={r['meta'].get('n_rebal', '--')}")

    print("\n" + "=" * 95)
    print("汇总对比")
    print("=" * 95)
    rows = []
    for r in results:
        jac = turnover_vs_baseline(r["blended_scores"], baseline_scores)
        rows.append({
            "配置": r["label"], "年化": f"{r['stats']['CAGR']*100:.1f}%",
            "夏普": f"{r['stats']['Sharpe']:.3f}", "回撤": f"{r['stats']['MaxDD']*100:.1f}%",
            "IS夏普": f"{r['stats_is']['Sharpe']:.3f}", "OOS夏普": f"{r['stats_oos']['Sharpe']:.3f}",
            "OOS/IS": f"{r['decay']:.2f}",
            "与纯1日Top3重合度": f"{jac:.3f}" if pd.notna(jac) else "--",
        })
    df = pd.DataFrame(rows).set_index("配置")
    print(df.to_string())

    sharpes = {r["label"]: r["stats"]["Sharpe"] for r in results}
    best = max(sharpes, key=sharpes.get)
    print(f"\n夏普最高：{best}（{sharpes[best]:.3f}）")

    # 分年度对比（首尾权重 + 中间0.5，避免输出过多）
    print("\n" + "=" * 95)
    print("分年度收益率对比（w=1.0基线 / w=0.5折中 / w=0.0纯上月25日）")
    print("=" * 95)
    sel = [r for r in results if r["label"] in ("w(本月1日)=1.0", "w(本月1日)=0.5", "w(本月1日)=0.0")]

    def yearly_returns(nav):
        yearly = {}
        for year, grp in nav.groupby(nav.index.year):
            if len(grp) < 2:
                continue
            yearly[year] = grp.iloc[-1] / grp.iloc[0] - 1
        return pd.Series(yearly)

    table = {r["label"]: yearly_returns(r["nav"]) for r in sel}
    ydf = pd.DataFrame(table).sort_index()
    ydf_fmt = ydf.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    print(ydf_fmt.to_string())


if __name__ == "__main__":
    main()
