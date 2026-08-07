"""
候选池构建规则优化 A-1：成交额门槛扫描（2026-07-27）

背景：45只手工标的池alpha已确认主要来自事后精选的强势赛道（见 research.md
"45只标的池行业构成分解测试"），机械化431只池（成交额>1亿元连续6月）夏普仅0.59，
用户要求探索更好的候选池构建规则。历史已验证"收窄/分层"类规则全部更差（申万分层
0.45/0.50，Top100上限0.53），本脚本测试反方向：单纯放宽流动性门槛、扩大候选池，
不引入任何行业/主题维度，看夏普是否随候选池扩大改善。

方法：复用 v23 的 point-in-time 候选池构建逻辑，只改 AMOUNT_THRESHOLD_QIAN，
扫描多个门槛（对应不同候选池规模），其余全部不变（动量窗口25日、风险调整、Top3等权）。
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
import etf_rotation_v23_universe_bias_test as v23  # noqa: E402
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH, build_daily_qualified, build_pit_universe,
    fetch_prices_for_candidates, load_close_matrix_from_cache,
    mask_scores_by_pit_universe,
)

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# 门槛梯度（单位：亿元），1.0对应当前上线机械化池
THRESHOLDS_YI = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def run_one_threshold(turnover: pd.DataFrame, threshold_yi: float) -> dict:
    v23.AMOUNT_THRESHOLD_QIAN = threshold_yi * 100_000
    amount_wide = build_daily_qualified(turnover)
    pit_universe = build_pit_universe(amount_wide)
    all_candidates = sorted(set().union(
        *pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))

    fetch_prices_for_candidates(all_candidates)

    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]

    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]

    nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF)
    stats = calc_stats(nav, f"门槛{threshold_yi}亿(候选{len(valid_codes)}只)")
    stats["候选池规模"] = len(valid_codes)
    stats["门槛(亿元)"] = threshold_yi
    return stats


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]

    rows = []
    for th in THRESHOLDS_YI:
        print(f"\n{'=' * 60}\n门槛 = {th} 亿元\n{'=' * 60}")
        stats = run_one_threshold(turnover, th)
        rows.append(stats)
        print(pd.DataFrame([stats]).set_index("标的").to_string())

    df = pd.DataFrame(rows).set_index("标的")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "v29_liquidity_threshold_sweep.csv"
    df.to_csv(out_path)

    print("\n" + "=" * 70)
    print("成交额门槛扫描汇总")
    print("=" * 70)
    print(df.to_string())
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
