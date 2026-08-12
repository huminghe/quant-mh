"""
候选池构建规则优化 B-1：同跟踪指数去重（2026-07-27）

背景：v29-v33（成交额门槛扫描/冲击成本敏感性/行业cap/PBO）已系统性证伪"候选池
扩容"方向。深度调研发现一个此前从未测试的规则维度：机械化431只候选池把
"多家基金公司发行同一跟踪指数ETF"（如11只中证A500ETF、9只恒生科技ETF）当独立
候选处理，这些标的收益几乎是同一份的复制品。Top3选股时同指数多只同时入选并不
构成真正分散，反而是v27诊断的"集中度风险"的另一重来源。

规则（与已证伪的申万行业分层性质不同——行业分层是粗粒度人工判断，按行业砍到
每类1-2只，会丧失同行业内部的轮动空间；本规则是精确、无歧义的事实性去重：
同一跟踪指数=同一收益源，不是"同行业不同暴露"）：
  按 fund_basic.benchmark 字段清洗出的核心指数名分组，每组只保留当月评估点
  滚动126日成交额最高（流动性最好）的一只作为代表，其余从候选池中剔除。

跟踪指数是ETF的静态属性（成立后基本不变），用当前 fetch_etf_benchmark.py 落盘
的快照做 point-in-time 近似不构成前视偏差——这是"跟踪哪个指数"这一事实，
不依赖任何未来收益/走势推断。

数据依赖：fetch_market_turnover.py（market_turnover.parquet/market_etf_meta.parquet）
         fetch_etf_benchmark.py（etf_benchmark.parquet）
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, START_DATE, CASH_ETF,
)
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH, LOOKBACK_TRADING_DAYS, MIN_VALID_DAYS, AMOUNT_THRESHOLD_QIAN,
    build_daily_qualified, fetch_prices_for_candidates, load_close_matrix_from_cache,
    mask_scores_by_pit_universe,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
BENCHMARK_PATH = DATA_DIR / "etf_benchmark.parquet"


def build_pit_universe_dedup(amount_wide: pd.DataFrame, benchmark_map: dict) -> pd.Series:
    """
    与 v23.build_pit_universe 相同的逐月评估逻辑，额外叠加：同一 benchmark_clean
    分组内，只保留当月评估点滚动均量最高的一只。没有 benchmark 映射的标的
    （不在 etf_benchmark.parquet 里，理论上不应发生，因为口径一致）视为独立分组，不受影响。
    """
    rolling_avg = amount_wide.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()
    qualified = rolling_avg > AMOUNT_THRESHOLD_QIAN

    ym = qualified.index.to_period("M")
    month_end_idx = qualified.groupby(ym).apply(lambda x: x.index[-1])

    pit = {}
    for period, eval_date in month_end_idx.items():
        row = qualified.loc[eval_date]
        codes = set(row[row].index)
        avg_row = rolling_avg.loc[eval_date]

        groups = {}
        for code in codes:
            bm = benchmark_map.get(code)
            key = bm if bm is not None else f"__standalone_{code}"
            groups.setdefault(key, []).append(code)

        kept = set()
        for key, members in groups.items():
            if len(members) == 1:
                kept.add(members[0])
            else:
                best = max(members, key=lambda c: avg_row.get(c, 0))
                kept.add(best)
        pit[period] = kept
    return pd.Series(pit)


def load_benchmark_map() -> dict:
    bm = pd.read_parquet(BENCHMARK_PATH)
    bm = bm.dropna(subset=["benchmark_clean"])
    return bm.set_index("ts_code")["benchmark_clean"].to_dict()


def split_in_out_sample(nav: pd.Series, frac: float = 0.8):
    """按时间顺序切样本内(前frac)/样本外(后1-frac)，各自独立算净值曲线（起点归一）。"""
    n = len(nav)
    cut = int(n * frac)
    nav_in = nav.iloc[:cut]
    nav_out = nav.iloc[cut - 1:]  # 从样本内最后一点接续，保证收益率连续
    nav_out = nav_out / nav_out.iloc[0] * nav.iloc[0]
    return nav_in, nav_out


def run_one(candidates_source: str, pit_universe: pd.Series, all_codes: list):
    fetch_prices_for_candidates(all_codes)
    close_full = load_close_matrix_from_cache(all_codes)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]

    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]

    nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF)

    label_full = f"{candidates_source}(候选{len(valid_codes)}只,全样本)"
    stats_full = calc_stats(nav, label_full)

    nav_in, nav_out = split_in_out_sample(nav, frac=0.8)
    stats_in = calc_stats(nav_in, f"{candidates_source}(样本内80%)")
    stats_out = calc_stats(nav_out, f"{candidates_source}(样本外20%)")

    return len(valid_codes), stats_full, stats_in, stats_out


def main():
    print("加载全市场成交额数据与跟踪指数映射...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    benchmark_map = load_benchmark_map()
    print(f"跟踪指数映射覆盖 {len(benchmark_map)} 只标的")

    amount_wide = build_daily_qualified(turnover)

    print("\n构建基线候选池（v23原始规则，无去重）...")
    from etf_rotation_v23_universe_bias_test import build_pit_universe as build_pit_baseline
    pit_baseline = build_pit_baseline(amount_wide)
    all_baseline = sorted(set().union(
        *pit_baseline.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))
    print(f"基线候选池历史累计标的数：{len(all_baseline)}")

    print("\n构建去重候选池（同跟踪指数只保留流动性最优一只）...")
    pit_dedup = build_pit_universe_dedup(amount_wide, benchmark_map)
    all_dedup = sorted(set().union(
        *pit_dedup.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))
    print(f"去重候选池历史累计标的数：{len(all_dedup)}")

    # 最新一期候选池规模对比（用于直观核对与探索阶段62%重复度一致）
    latest_period = pit_baseline.index.max()
    print(f"\n最新评估期({latest_period})：基线{len(pit_baseline[latest_period])}只 "
          f"→ 去重后{len(pit_dedup[latest_period])}只")

    results = {}
    for label, pit_universe, all_codes in [
        ("基线(无去重)", pit_baseline, all_baseline),
        ("去重(同指数留1只)", pit_dedup, all_dedup),
    ]:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        n_valid, stats_full, stats_in, stats_out = run_one(label, pit_universe, all_codes)
        results[label] = dict(n_valid=n_valid, full=stats_full, in_=stats_in, out=stats_out)
        for tag, s in [("全样本", stats_full), ("样本内80%", stats_in), ("样本外20%", stats_out)]:
            print(f"\n[{tag}]")
            for k, v in s.items():
                if k != "标的":
                    print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    rows = []
    for label, r in results.items():
        for tag, s in [("全样本", r["full"]), ("样本内80%", r["in_"]), ("样本外20%", r["out"])]:
            row = dict(s)
            row["配置"] = label
            row["区间"] = tag
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["配置", "区间"])
    print(df[["总收益", "年化收益(CAGR)", "年化夏普", "最大回撤", "年化波动率", "Calmar"]].to_string())

    # 样本外/样本内夏普比值，按 trading-standards.md 阈值判断是否可能过拟合
    for label, r in results.items():
        sharpe_in = float(r["in_"]["年化夏普"])
        sharpe_out = float(r["out"]["年化夏普"])
        ratio = sharpe_out / sharpe_in if sharpe_in != 0 else float("nan")
        flag = "（样本外<样本内*0.5，可能过拟合）" if sharpe_in > 0 and sharpe_out < sharpe_in * 0.5 else ""
        print(f"\n{label}：样本外/样本内夏普比 = {ratio:.2f}{flag}")


if __name__ == "__main__":
    main()
