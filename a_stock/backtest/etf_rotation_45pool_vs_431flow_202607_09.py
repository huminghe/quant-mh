"""
45池（纯风险调整动量）vs 431池+flow（线上配置）反事实对比，延伸至2026-09-23

背景：2026-08-24曾算过一次反事实对比（07-01→08-31），本次用户要求延伸到最新
日期。431+flow自2026-08-12上线后signal_log.csv没有真实模拟盘记录（最后一条
真实记录是08-11，切换前的45池数据；09月第一个交易日也没人手动跑信号），
所以整个07-01→09-23区间对45池和431+flow两条路径都只能用回测脚本反算，
不是真实模拟盘数据对比，是纯回测。

45池：etf_universe.ETF_UNIVERSE，纯风险调整动量（历史上线配置，从未叠加flow）
431+flow：etf_all_candidates.parquet 431只候选池 + flow信号连续打折
  （day_scores *= (0.5 + boost)，boost=1-rank(pct=True)，与v38/signal_today.py一致）
基准：510300.SH 沪深300ETF

两条路径均用同一套run_backtest（T+1建仓，佣金万1+滑点万2双边），Top3等权、月度调仓。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro  # noqa: E402
from etf_universe import ETF_CODES  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all  # noqa: E402
from etf_rotation_v18_signal_ablation import (  # noqa: E402
    calc_risk_adj_momentum, run_backtest, calc_stats, get_rebalance_dates,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
CANDIDATES_FILE = DATA_DIR / "etf_all_candidates.parquet"
BENCHMARK_CODE = "510300.SH"

FULL_START = "2016-01-01"   # 计算动量/flow需要的历史窗口起点（不是展示窗口）
WINDOW_START = "2026-06-26"  # 展示窗口起点，对齐此前45池模拟盘启动日
WINDOW_END = "2026-09-23"


def build_431_flow_scores():
    codes = pd.read_parquet(CANDIDATES_FILE)["ts_code"].tolist()
    close_full = load_close_matrix(codes + [BENCHMARK_CODE])
    close_full = close_full[close_full.index >= FULL_START]
    valid_codes = [c for c in codes if c in close_full.columns and close_full[c].notna().sum() >= 45]
    close = close_full[valid_codes]
    print(f"431池有效标的：{len(valid_codes)} 只，价格区间至 {close.index[-1].date()}")

    scores = calc_risk_adj_momentum(close)[valid_codes]
    rebal_dates = get_rebalance_dates(close.index)

    print("拉取431池ETF份额数据（flow信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=FULL_START.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    boost_rows = {}
    for d in rebal_dates:
        if flow_1m.empty:
            continue
        idx = flow_1m.index[flow_1m.index <= d]
        if len(idx) == 0:
            continue
        flow_d = flow_1m.loc[idx[-1]].dropna()
        if len(flow_d) < 5:
            continue
        r = flow_d.rank(pct=True)
        boost_rows[d] = 1 - r
    boost_signal = pd.DataFrame(boost_rows).T
    boost_signal.index = pd.to_datetime(boost_signal.index)

    return close, scores, rebal_dates, boost_signal


def build_45pool_scores():
    close_full = load_close_matrix(ETF_CODES + [BENCHMARK_CODE])
    close_full = close_full[close_full.index >= FULL_START]
    valid_codes = [c for c in ETF_CODES if c in close_full.columns]
    close = close_full[valid_codes]
    print(f"45池有效标的：{len(valid_codes)} 只，价格区间至 {close.index[-1].date()}")

    scores = calc_risk_adj_momentum(close)[valid_codes]
    rebal_dates = get_rebalance_dates(close.index)
    return close, scores, rebal_dates


def slice_window(nav: pd.Series) -> pd.Series:
    return nav[(nav.index >= WINDOW_START) & (nav.index <= WINDOW_END)]


def window_stats(nav: pd.Series) -> dict:
    nav = slice_window(nav)
    if len(nav) < 2:
        return {"总收益": None, "年化夏普": None, "最大回撤": None}
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    stats = calc_stats(nav)
    return {"总收益": total_ret, "年化夏普": stats["Sharpe"], "最大回撤": stats["MaxDD"]}


def main():
    print("=" * 60)
    print("构建431池+flow净值（线上配置回测反算）...")
    close_431, scores_431, rebal_431, boost_431 = build_431_flow_scores()
    nav_431 = run_backtest(close_431, scores_431, rebal_431, boost_signal=boost_431, boost_mode="continuous")

    print("=" * 60)
    print("构建45池净值（纯动量，历史上线配置反事实反算）...")
    close_45, scores_45, rebal_45 = build_45pool_scores()
    nav_45 = run_backtest(close_45, scores_45, rebal_45)

    print("=" * 60)
    print("沪深300基准...")
    bench_close = load_close_matrix([BENCHMARK_CODE])[BENCHMARK_CODE]
    bench_nav = slice_window(bench_close)
    bench_ret = bench_nav.iloc[-1] / bench_nav.iloc[0] - 1

    print()
    print(f"窗口：{WINDOW_START} → {WINDOW_END}（45池、431+flow均为事后回测反算，非真实模拟盘记录）")
    print()
    for name, nav in [("45池（假设情形，纯动量）", nav_45), ("431+flow（线上配置）", nav_431)]:
        s = window_stats(nav)
        print(f"{name}：总收益 {s['总收益']:.2%}，年化夏普 {s['年化夏普']:.3f}，最大回撤 {s['最大回撤']:.2%}")
    print(f"沪深300ETF基准：总收益 {bench_ret:.2%}")

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame({
        "date": slice_window(nav_45).index,
        "nav_45pool": slice_window(nav_45).values,
    }).set_index("date")
    df2 = pd.DataFrame({"nav_431flow": slice_window(nav_431).values}, index=slice_window(nav_431).index)
    df3 = pd.DataFrame({"nav_hs300": bench_nav.values}, index=bench_nav.index)
    merged = df.join(df2, how="outer").join(df3, how="outer")
    out_path = out_dir / "45pool_vs_431flow_202607_09_navs.csv"
    merged.to_csv(out_path)
    print(f"\n净值序列已保存：{out_path}")


if __name__ == "__main__":
    main()
