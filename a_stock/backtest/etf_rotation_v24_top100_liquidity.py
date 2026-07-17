"""
标的池选择偏差实测——达标+Top100上限变体（2026-07-15）

背景：v23（etf_rotation_v23_universe_bias_test.py）用固定阈值（滚动126日日均成交额
>1亿元）构建机械化候选池，431只候选，夏普0.59，明显低于生产标的池（45只手工圈定）
的1.053。用户追问：每次调仓时按成交额排名只取Top100，结果会怎样？

最初版本（已废弃）强制凑够100只——早期市场总量小、达标数不足100只时也会拿排名
靠后的低流动性标的凑数。用户指出不应强制凑数：**未达标时有多少只算多少只**
（如2018年只有24只过阈值，就只用24只），Top100只作为达标者数量过多时的上限。

与v23的区别：v23候选池规模随市场流动性单调扩容、无上限（2026年已达408只，
可能引入较多边际标的稀释信号）；v24（本脚本）在"过阈值"的基础上叠加"最多
100只"的上限，规模较大的年份从阈值集合里按成交额取前100，其余年份与v23完全一致。

复用v23的候选池数据加载/复权价格拉取/掩码回测逻辑，只新增按排名筛选的函数。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, BENCHMARK, START_DATE, CASH_ETF,
)
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    DATA_DIR, TURNOVER_PATH, META_PATH,
    LOOKBACK_TRADING_DAYS, MIN_VALID_DAYS, AMOUNT_THRESHOLD_QIAN,
    build_daily_qualified, fetch_prices_for_candidates,
    load_close_matrix_from_cache, mask_scores_by_pit_universe,
)

TOP_N_LIQUIDITY = 100


def build_pit_universe_capped(
    amount_wide: pd.DataFrame,
    threshold: float = AMOUNT_THRESHOLD_QIAN,
    top_n: int = TOP_N_LIQUIDITY,
) -> pd.Series:
    """
    在v23"滚动126日日均成交额>阈值"的达标集合基础上，叠加数量上限：
    达标数 <= top_n 时与v23完全一致（不强制凑数）；达标数 > top_n 时
    按成交额从达标集合里截取前 top_n 只。
    """
    rolling_avg = amount_wide.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()

    ym = rolling_avg.index.to_period("M")
    month_end_idx = rolling_avg.groupby(ym).apply(lambda x: x.index[-1])

    pit = {}
    for period, eval_date in month_end_idx.items():
        row = rolling_avg.loc[eval_date].dropna()
        qualified = row[row > threshold]
        if len(qualified) > top_n:
            qualified = qualified.sort_values(ascending=False).head(top_n)
        pit[period] = set(qualified.index)
    return pd.Series(pit)


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    print(f"成交额记录 {len(turnover)} 行，ETF基础信息 {len(meta)} 只")

    print("构建逐月达标+Top100上限候选池...")
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    print(f"过滤到股票型ETF后成交额记录 {len(turnover)} 行")
    amount_wide = build_daily_qualified(turnover)
    pit_universe = build_pit_universe_capped(amount_wide, AMOUNT_THRESHOLD_QIAN, TOP_N_LIQUIDITY)

    sizes = pit_universe.apply(len)
    print(f"逐月候选池规模范围：{sizes.min()}～{sizes.max()}只（被上限截断的月份数：{(sizes >= TOP_N_LIQUIDITY).sum()}/{len(sizes)}）")

    all_candidates = sorted(set().union(*pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())))
    print(f"历史上任一时点曾入选的ETF共 {len(all_candidates)} 只")

    print("拉取候选池复权价格（复用v23缓存，已缓存的跳过）...")
    fetch_prices_for_candidates(all_candidates)

    print("加载价格矩阵...")
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]
    print(f"有效标的数：{len(valid_codes)}（历史上任一时点曾入选，vs 生产标的池45只）")

    print(f"计算动量得分（窗口={MOMENTUM_WINDOW}日）...")
    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)

    print("按逐月候选池掩码得分...")
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    print(f"调仓日数量：{len(rebal_dates)}")

    print("运行回测（达标+Top100上限候选池）...")
    nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF)

    bench = close[BENCHMARK].dropna() if BENCHMARK in close.columns else None
    print("\n" + "=" * 60)
    print("标的池选择偏差实测结果——达标+Top100上限变体")
    print("=" * 60)
    print(f"回测区间：{nav.index[0].date()} → {nav.index[-1].date()}")
    stats = calc_stats(nav, f"达标且Top{TOP_N_LIQUIDITY}上限候选池")
    print(pd.DataFrame([stats]).set_index("标的").to_string())
    print("\n对照：v23固定阈值无上限候选池(431只)夏普0.59；生产标的池(45只手工圈定)全样本夏普1.053")


if __name__ == "__main__":
    main()
