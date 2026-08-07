"""
0.05亿门槛方案冲击成本敏感性测试（2026-07-27）

背景：v29门槛扫描（修复缓存bug后）显示门槛越松候选池越大夏普越高，0.05亿门槛
（500万日均成交额，1312只候选池）夏普0.76/回撤-24.2%显著优于当前1.0亿门槛
（431只）的0.53/-38.0%。但该回测未叠加任何冲击成本模型——纯用收盘价成交，
对百万级资金而言，交易日均成交额仅500万元的标的，冲击成本可能不可忽略。

本脚本用简化线性冲击成本模型做敏感性测试，验证0.05亿门槛的优势能否扛住冲击成本：
  单边冲击成本 = impact_coef * (交易金额 / 当日成交额)，叠加在原有滑点(万2双边)之上。
  这是粗略的线性近似（非Almgren-Chriss平方根模型），只用于判断"优势是否稳健"的
  方向性问题，不是精确的成本估计。

对比1.0亿门槛（431只，当前生产池）、0.05亿门槛（1312只，v29最优方案）以及
0.1亿/0.2亿门槛（折中方案）在impact_coef=0/0.01/0.05/0.1/0.2/0.5 六档下的表现。
impact_coef=0等价于v29原始结果。

第一轮结果（仅1.0亿vs0.05亿）已确认：0.05亿门槛的夏普优势(0.72 vs 0.49)在
impact_coef>=0.1时基本消失，>=0.2时大幅劣于1.0亿门槛，说明该优势主要来自零冲击
成本假设，不稳健。本轮补测0.1亿/0.2亿门槛，判断是否存在更抗冲击成本的折中门槛。
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

THRESHOLDS_YI = [1.0, 0.05, 0.1, 0.2]  # 当前生产池 vs v29门槛扫描候选方案
IMPACT_COEFS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]


def build_pool(turnover: pd.DataFrame, threshold_yi: float):
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
    return close, masked_scores, rebal_dates, amount_wide, len(valid_codes)


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]

    rows = []
    for th in THRESHOLDS_YI:
        print(f"\n{'=' * 70}\n构建门槛 {th} 亿元候选池...\n{'=' * 70}")
        close, masked_scores, rebal_dates, amount_wide, pool_size = build_pool(turnover, th)
        print(f"候选池规模：{pool_size}只，调仓日数量：{len(rebal_dates)}")

        for coef in IMPACT_COEFS:
            nav = run_backtest(
                close, masked_scores, rebal_dates, cash_etf=CASH_ETF,
                amount_wide=amount_wide, impact_coef=coef,
            )
            label = f"门槛{th}亿(候选{pool_size}只)_impact={coef}"
            stats = calc_stats(nav, label)
            stats["门槛(亿元)"] = th
            stats["候选池规模"] = pool_size
            stats["impact_coef"] = coef
            rows.append(stats)
            print(f"  impact_coef={coef}: 夏普={stats['年化夏普']}, 回撤={stats['最大回撤']}")

    df = pd.DataFrame(rows).set_index("标的")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "v31_impact_cost_sensitivity.csv"
    df.to_csv(out_path)

    print("\n" + "=" * 70)
    print("冲击成本敏感性汇总")
    print("=" * 70)
    print(df.to_string())
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
