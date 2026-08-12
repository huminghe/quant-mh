"""
候选池构建规则优化 B-3 续：综合流动性排名 drop_frac 敏感性扫描（2026-07-27）

背景：v35验证"成交额门槛+Amihud/Corwin-Schultz综合排名砍尾20%"全维度小幅改善
（全样本夏普0.49→0.53，最大回撤-38.0%→-35.3%），但幅度不大，需要判断这是真实
alpha还是对20%这个特定切点的过拟合。本脚本复用v35的流动性代理计算，扫描
drop_frac ∈ {0%(基线), 10%, 20%, 30%, 40%}，看夏普/回撤是否随砍尾比例单调变化——
单调且稳健支持"综合流动性排名是真信号"，忽高忽低则更像噪音。

数据依赖：与v35相同，复用其Amihud/Corwin-Schultz计算与候选池构建函数。
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, START_DATE, CASH_ETF,
)
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH, build_daily_qualified, build_pit_universe,
    fetch_prices_for_candidates, load_close_matrix_from_cache, mask_scores_by_pit_universe,
)
from etf_rotation_v34_dedup_by_benchmark import split_in_out_sample  # noqa: E402
from etf_rotation_v35_liquidity_composite import (  # noqa: E402
    load_hl_matrix_from_cache, calc_amihud_daily, calc_corwin_schultz_daily,
    build_pit_universe_liquidity_rank,
)

DROP_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4]


def run_one(label: str, pit_universe: pd.Series, all_codes: list):
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

    stats_full = calc_stats(nav, f"{label}(候选{len(valid_codes)}只,全样本)")
    nav_in, nav_out = split_in_out_sample(nav, frac=0.8)
    stats_in = calc_stats(nav_in, f"{label}(样本内80%)")
    stats_out = calc_stats(nav_out, f"{label}(样本外20%)")

    return len(valid_codes), stats_full, stats_in, stats_out


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    amount_wide = build_daily_qualified(turnover)

    print("构建基线候选池（v23原始规则，仅成交额门槛）...")
    pit_baseline = build_pit_universe(amount_wide)
    all_baseline = sorted(set().union(
        *pit_baseline.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))
    print(f"基线候选池历史累计标的数：{len(all_baseline)}")

    print("拉取/确认候选价格缓存（含OHLC）...")
    fetch_prices_for_candidates(all_baseline)

    print("加载OHLC矩阵，计算Amihud与Corwin-Schultz流动性代理（全脚本只算一次）...")
    close_full = load_close_matrix_from_cache(all_baseline)
    high_full, low_full = load_hl_matrix_from_cache(all_baseline)
    amihud_daily = calc_amihud_daily(close_full, amount_wide)
    cs_daily = calc_corwin_schultz_daily(high_full, low_full)

    results = {}
    for drop_frac in DROP_FRACS:
        if drop_frac == 0.0:
            label = "drop_frac=0%(基线)"
            pit_universe = pit_baseline
            all_codes = all_baseline
        else:
            label = f"drop_frac={drop_frac:.0%}"
            pit_universe = build_pit_universe_liquidity_rank(
                amount_wide, amihud_daily, cs_daily, drop_frac=drop_frac
            )
            all_codes = sorted(set().union(
                *pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())
            ))

        latest_period = pit_universe.index.max()
        print(f"\n{'=' * 70}\n{label}（历史累计{len(all_codes)}只，最新期{len(pit_universe[latest_period])}只）\n{'=' * 70}")
        n_valid, stats_full, stats_in, stats_out = run_one(label, pit_universe, all_codes)
        results[label] = dict(n_valid=n_valid, full=stats_full, in_=stats_in, out=stats_out)

    print("\n" + "=" * 70)
    print("drop_frac 敏感性汇总")
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

    print("\n全样本夏普 vs drop_frac（判断单调性）：")
    for label, r in results.items():
        print(f"  {label}: 夏普={r['full']['年化夏普']}, 最大回撤={r['full']['最大回撤']}, 候选数={r['n_valid']}")


if __name__ == "__main__":
    main()
