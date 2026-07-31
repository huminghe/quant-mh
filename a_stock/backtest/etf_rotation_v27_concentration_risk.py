"""
Top3 行业集中度风险历史诊断（机械化候选池版）

背景：模拟盘复核发现 06-26~07-24 段 Top3 持仓恰好全部集中在同一泛半导体/科技赛道，
组合等权相当于集中押注同一方向，跑输基准14pp（详见 docs/research.md"模拟盘实盘表现复核"）。
用户方法论决策（2026-07-27）：所有策略测试基准改为机械化候选池（v23，431只，纯流动性
规则，point-in-time构建），不再用45只手工池验证任何改进。

本脚本目的：在机械化候选池上跑一遍历史全样本（不改选股逻辑，只做事后归因统计），量化：
  1. 每次调仓时 Top3 持仓的行业集中度分布（3只同行业 / 2只同行业 / 分散）
  2. 集中月份 vs 分散月份，随后一个月的组合收益差异（等权、不含成本，直接归因）
  3. 用价格相关性矩阵（选股当日回溯60日）做交叉验证，不完全依赖行业分类口径

行业口径：复用 etf_sw_exposure.parquet 的 dominant_industry 字段（申万一级 + 港股
同花顺，07-27 已修复港股匹配率0%的bug）。不产生新数据文件，只读取已有缓存。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_all_scores, get_rebalance_dates, MOMENTUM_WINDOW, TOP_N, START_DATE  # noqa: E402
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH,
    build_daily_qualified, build_pit_universe,
    load_close_matrix_from_cache, mask_scores_by_pit_universe,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_PATH = DATA_DIR / "etf_sw_exposure.parquet"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
CORR_WINDOW = 60  # 与 etf_rotation.py 的相关性过滤窗口一致，便于交叉验证


def build_universe_close_and_scores():
    """复用 v23 的候选池构建 + 打分逻辑，返回 (close, masked_scores, rebal_dates)。"""
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]

    amount_wide = build_daily_qualified(turnover)
    pit_universe = build_pit_universe(amount_wide)
    all_candidates = sorted(set().union(
        *pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))

    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]

    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    return close, masked_scores, rebal_dates


def classify_concentration(top3: list, industry_map: dict) -> str:
    """按 Top3 持仓的 dominant_industry 判定集中度等级。"""
    industries = [industry_map.get(c) for c in top3]
    valid = [i for i in industries if pd.notna(i)]
    if not valid:
        return "分散"
    counts = pd.Series(valid).value_counts()
    max_group = counts.iloc[0]
    if max_group >= 3:
        return "3/3同行业"
    if max_group == 2:
        return "2/3同行业"
    return "分散"


def avg_pairwise_corr(close: pd.DataFrame, top3: list, date: pd.Timestamp) -> float:
    """选股当日回溯 CORR_WINDOW 日的日收益率相关性，Top3 两两平均。"""
    if len(top3) < 2:
        return np.nan
    date_loc = close.index.get_loc(date)
    window_start = max(0, date_loc - CORR_WINDOW)
    ret_window = close.iloc[window_start:date_loc][top3].pct_change().dropna()
    if len(ret_window) < CORR_WINDOW // 2:
        return np.nan
    corr = ret_window.corr()
    pairs = []
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            pairs.append(corr.iloc[i, j])
    return float(np.nanmean(pairs)) if pairs else np.nan


def main():
    print("重建机械化候选池、价格矩阵、动量得分...")
    close, masked_scores, rebal_dates = build_universe_close_and_scores()
    print(f"候选池标的数：{close.shape[1]}，调仓日数量：{len(rebal_dates)}")

    exposure = pd.read_parquet(EXPOSURE_PATH)
    industry_map = exposure.set_index("ts_code")["dominant_industry"].to_dict()

    records = []
    for i, date in enumerate(rebal_dates):
        day_scores = masked_scores.loc[date].dropna()
        pos_scores = day_scores[day_scores > 0].nlargest(TOP_N)
        top3 = list(pos_scores.index)
        if len(top3) < TOP_N:
            continue  # 候选不足3只（含空仓月），不纳入集中度统计

        level = classify_concentration(top3, industry_map)
        corr = avg_pairwise_corr(close, top3, date)

        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else close.index[-1]
        rets = []
        for code in top3:
            p0, p1 = close.loc[date, code], close.loc[next_date, code]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append(p1 / p0 - 1)
        fwd_return = float(np.mean(rets)) if rets else np.nan

        records.append({
            "date": date, "top3": ",".join(top3), "level": level,
            "avg_pairwise_corr": corr, "fwd_return": fwd_return,
        })

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "v27_concentration_risk.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("集中度等级分布 + 后续一月收益统计")
    print("=" * 70)
    summary = df.groupby("level")["fwd_return"].agg(
        月份数="count", 平均收益="mean", 收益中位数="median",
        最差收益="min", 负收益占比=lambda x: (x < 0).mean(),
    )
    summary["平均收益"] = (summary["平均收益"] * 100).round(2).astype(str) + "%"
    summary["收益中位数"] = (summary["收益中位数"] * 100).round(2).astype(str) + "%"
    summary["最差收益"] = (summary["最差收益"] * 100).round(2).astype(str) + "%"
    summary["负收益占比"] = (summary["负收益占比"] * 100).round(1).astype(str) + "%"
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("交叉验证：按选股当日Top3两两平均相关性分桶")
    print("=" * 70)
    df_corr = df.dropna(subset=["avg_pairwise_corr"]).copy()
    df_corr["corr_bucket"] = pd.cut(
        df_corr["avg_pairwise_corr"], bins=[-1, 0.3, 0.7, 1.0],
        labels=["低相关(<0.3)", "中相关(0.3-0.7)", "高相关(>0.7)"],
    )
    summary_corr = df_corr.groupby("corr_bucket")["fwd_return"].agg(
        月份数="count", 平均收益="mean", 最差收益="min",
        负收益占比=lambda x: (x < 0).mean(),
    )
    summary_corr["平均收益"] = (summary_corr["平均收益"] * 100).round(2).astype(str) + "%"
    summary_corr["最差收益"] = (summary_corr["最差收益"] * 100).round(2).astype(str) + "%"
    summary_corr["负收益占比"] = (summary_corr["负收益占比"] * 100).round(1).astype(str) + "%"
    print(summary_corr.to_string())

    total = len(df)
    concentrated = df["level"].isin(["3/3同行业", "2/3同行业"]).sum()
    print(f"\n全样本 {total} 个调仓月，集中（2/3或3/3同行业）占比："
          f"{concentrated}/{total} = {concentrated / total * 100:.1f}%")
    print(f"明细已保存至 {out_path}")


if __name__ == "__main__":
    main()
