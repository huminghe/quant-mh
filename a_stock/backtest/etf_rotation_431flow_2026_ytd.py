"""
431池+flow（线上配置）2026年至今整体回测

背景：模拟盘2026-08-12才上线，之前（01-01→08-11）没有真实运行记录。
本脚本用回测反算整个2026年至今的表现，把线上配置当成全年都在跑来看
整体回测情况（不是模拟盘真实数据，是纯回测反算）。

431+flow：etf_all_candidates.parquet 431只候选池 + flow信号连续打折
（day_scores *= (0.5 + boost)，boost=1-rank(pct=True)，与signal_today.py一致）
基准：510300.SH 沪深300ETF
T+1建仓，佣金万1+滑点万2双边，Top3等权、月度调仓。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all  # noqa: E402
from etf_rotation_v18_signal_ablation import (  # noqa: E402
    calc_risk_adj_momentum, run_backtest, calc_stats, get_rebalance_dates,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
CANDIDATES_FILE = DATA_DIR / "etf_all_candidates.parquet"
BENCHMARK_CODE = "510300.SH"

FULL_START = "2016-01-01"   # 计算动量/flow需要的历史窗口起点（不是展示窗口）
WINDOW_START = "2026-01-01"
WINDOW_END = None  # None = 取到最新可用数据


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


def slice_window(nav: pd.Series, end: str = None) -> pd.Series:
    end_date = pd.Timestamp(end) if end else nav.index[-1]
    return nav[(nav.index >= WINDOW_START) & (nav.index <= end_date)]


def window_stats(nav: pd.Series, end: str = None) -> dict:
    nav = slice_window(nav, end)
    if len(nav) < 2:
        return {"总收益": None, "年化夏普": None, "最大回撤": None}
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    stats = calc_stats(nav)
    return {"总收益": total_ret, "年化夏普": stats["Sharpe"], "最大回撤": stats["MaxDD"]}


def monthly_breakdown(nav: pd.Series, bench: pd.Series) -> pd.DataFrame:
    nav_m = nav.resample("ME").last()
    bench_m = bench.resample("ME").last()
    nav_ret = nav_m.pct_change()
    bench_ret = bench_m.pct_change()
    df = pd.DataFrame({"策略月收益": nav_ret, "沪深300月收益": bench_ret})
    df["超额"] = df["策略月收益"] - df["沪深300月收益"]
    return df.dropna(how="all")


def main():
    print("=" * 60)
    print("构建431池+flow净值（线上配置，2026年至今回测反算）...")
    close_431, scores_431, rebal_431, boost_431 = build_431_flow_scores()
    nav_431 = run_backtest(close_431, scores_431, rebal_431, boost_signal=boost_431, boost_mode="continuous")

    print("=" * 60)
    print("沪深300基准...")
    bench_close = load_close_matrix([BENCHMARK_CODE])[BENCHMARK_CODE]

    end_date = min(nav_431.index[-1], bench_close.index[-1])
    nav_w = slice_window(nav_431, end_date)
    bench_w = slice_window(bench_close, end_date)
    bench_ret = bench_w.iloc[-1] / bench_w.iloc[0] - 1

    print()
    print(f"窗口：{WINDOW_START} → {end_date.date()}（431+flow为事后回测反算，非真实模拟盘记录）")
    print()
    s = window_stats(nav_431, end_date)
    print(f"431+flow（线上配置）：总收益 {s['总收益']:.2%}，年化夏普 {s['年化夏普']:.3f}，最大回撤 {s['最大回撤']:.2%}")
    print(f"沪深300ETF基准：总收益 {bench_ret:.2%}")
    print(f"超额收益：{s['总收益'] - bench_ret:+.2%}")

    print()
    print("── 月度拆分 " + "─" * 40)
    mb = monthly_breakdown(nav_w, bench_w)
    for idx, row in mb.iterrows():
        print(f"  {idx.strftime('%Y-%m')}  策略 {row['策略月收益']:+.2%}  沪深300 {row['沪深300月收益']:+.2%}  超额 {row['超额']:+.2%}")

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "etf_rotation_431flow_2026_ytd_navs.csv"
    merged = pd.DataFrame({"nav_431flow": nav_w, "nav_hs300": bench_w}).dropna(how="all")
    merged.to_csv(out_path)
    print(f"\n净值序列已保存：{out_path}")


if __name__ == "__main__":
    main()
