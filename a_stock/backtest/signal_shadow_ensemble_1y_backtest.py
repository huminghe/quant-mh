"""
ML弱信号集成影子监控 —— 近一年回测对比

背景：`signal_shadow_ensemble.py` 从2026-07-14开始逐月记录线上版（纯动量）
与集成版（动量+拥挤度/成交量确认/资金流打折）的持仓分歧，但实盘刚跑一次，
样本不足以判断。本脚本用历史数据回填近一年的月度持仓对比，并跑两版组合
净值曲线，作为观察期内的补充参考（不改变"观察1-2个季度再决定"的既定计划，
详见 a_stock/docs/research.md ML弱信号集成影子监控小节）。

信号计算逻辑与`etf_rotation_v16_signal_combo_ablation.py`完全一致（直接
复用其函数），只是把回测/对比窗口限定在最近约12个月。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch_data import load_close_matrix, init_pro
from etf_universe import ETF_UNIVERSE
from etf_rotation_v16_signal_combo_ablation import (
    calc_risk_adj_momentum, calc_crowding, load_amount_matrix, fetch_fund_share_all,
    get_rebalance_dates, run_backtest, calc_stats, MOMENTUM_WINDOW, TOP_N,
)

LOOKBACK_YEARS = 1
RESULT_CSV = pathlib.Path(__file__).parent / "results" / "shadow_1y_monthly_compare.csv"


def main():
    print("加载价格与成交额数据...")
    close_full = load_close_matrix()
    valid_codes = [c for c in close_full.columns if close_full[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close_full = close_full[valid_codes]

    cutoff = close_full.index[-1] - pd.DateOffset(years=LOOKBACK_YEARS)
    close_1y = close_full[close_full.index >= cutoff]
    print(f"近{LOOKBACK_YEARS}年窗口：{close_1y.index[0].date()} ~ {close_1y.index[-1].date()}")

    print("计算风险调整动量（全历史算，保证窗口起点lookback完整）...")
    scores_full = calc_risk_adj_momentum(close_full)[valid_codes]
    scores_1y = scores_full[scores_full.index >= cutoff]

    print("计算拥挤度信号（全历史滚动相关性，较慢）...")
    crowding_full = calc_crowding(close_full)

    print("计算成交量确认信号...")
    amount = load_amount_matrix()
    amount = amount[[c for c in valid_codes if c in amount.columns]]
    vol_ratio_full = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取ETF份额数据（资金流信号，近一年+缓冲）...")
    pro = init_pro()
    flow_start = (cutoff - pd.Timedelta(days=60)).strftime("%Y%m%d")
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=flow_start)
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m_full = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    rebal_dates = [d for d in get_rebalance_dates(close_1y.index)]

    # ── 逐月持仓对比 ──────────────────────────────────────

    def combo_rank_at(d):
        crowd_d = crowding_full.loc[d] if d in crowding_full.index else pd.Series(dtype=float)
        volr_d = vol_ratio_full.loc[d] if d in vol_ratio_full.index else pd.Series(dtype=float)
        flow_d = pd.Series(dtype=float)
        if not flow_1m_full.empty:
            idx = flow_1m_full.index[flow_1m_full.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m_full.loc[idx[-1]]
        ranks = []
        for s, invert in [(crowd_d, True), (volr_d, False), (flow_d, True)]:
            s = s.dropna()
            if len(s) < 5:
                continue
            r = s.rank(pct=True)
            if invert:
                r = 1 - r
            ranks.append(r)
        if not ranks:
            return pd.Series(dtype=float)
        return pd.concat(ranks, axis=1).mean(axis=1)

    print("\n" + "=" * 90)
    print(f"逐月持仓对比（近{LOOKBACK_YEARS}年，共{len(rebal_dates)}个月）")
    print("=" * 90)

    rows = []
    diff_months = 0
    for d in rebal_dates:
        online_scores = scores_1y.loc[d].dropna()
        online_target = list(online_scores[online_scores > 0].nlargest(TOP_N).index)

        boost = combo_rank_at(d)
        ens_scores = online_scores.copy()
        for code in ens_scores.index:
            if code in boost.index and not pd.isna(boost[code]):
                ens_scores[code] *= (0.5 + boost[code])
        ensemble_target = list(ens_scores[ens_scores > 0].nlargest(TOP_N).index)

        same = online_target == ensemble_target
        if not same:
            diff_months += 1
        rows.append({
            "日期": d.date(), "线上持仓": "/".join(online_target) or "现金",
            "集成持仓": "/".join(ensemble_target) or "现金", "一致": same,
        })
        marker = "" if same else "  ← 分歧"
        print(f"  {d.date()}  线上: {'/'.join(online_target) or '现金':<30}  "
              f"集成: {'/'.join(ensemble_target) or '现金':<30}{marker}")

    print(f"\n{len(rebal_dates)}个月中，{diff_months}个月出现持仓分歧（{diff_months/len(rebal_dates):.1%}）")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
    print(f"逐月对比已保存：{RESULT_CSV}")

    # ── 组合层面回测对比 ──────────────────────────────────

    print("\n" + "=" * 90)
    print(f"组合净值回测对比（近{LOOKBACK_YEARS}年）")
    print("=" * 90)

    boost_full = pd.DataFrame({d: combo_rank_at(d) for d in rebal_dates}).T

    nav_online = run_backtest(close_1y, scores_1y, rebal_dates)
    nav_ensemble = run_backtest(close_1y, scores_1y, rebal_dates, boost_signal=boost_full, boost_mode="continuous")

    stats_online = calc_stats(nav_online)
    stats_ensemble = calc_stats(nav_ensemble)

    print(f"\n{'':<12}{'年化':>10}{'夏普':>10}{'最大回撤':>10}")
    print(f"{'线上版':<12}{stats_online['CAGR']*100:>9.1f}%{stats_online['Sharpe']:>10.3f}{stats_online['MaxDD']*100:>9.1f}%")
    print(f"{'集成版':<12}{stats_ensemble['CAGR']*100:>9.1f}%{stats_ensemble['Sharpe']:>10.3f}{stats_ensemble['MaxDD']*100:>9.1f}%")
    print(f"\nΔ夏普 = {stats_ensemble['Sharpe'] - stats_online['Sharpe']:+.3f}")
    print("\n注：近1年仅约12个调仓点，单一年度样本量太小，此结果仅供观察期参考，不构成独立结论。")
    print("=" * 90)


if __name__ == "__main__":
    main()
