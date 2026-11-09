"""
加密货币市场趋势性/波动性逐月走势（最近N个月）

背景：2026-11-04 market_trend_volatility_check.py 用"近期vs历史"二分对比确认了
波动率收缩跨周期一致、趋势性仅中短周期(4h/1h)显著减弱。本脚本把近期窗口按自然月
拆开，看这两个现象是从哪个月开始、是否逐月加剧，而不是只看一个笼统的"近N天"均值。

默认跑1d/8h/4h/1h四档（与 market_trend_volatility_check.py 的 --timeframes 覆盖范围一致）。
复用 market_trend_volatility_check.py 的拉取/指标函数。

2026-11-09方法论改进：每月除了原始绝对值（年化波动率%、ADX、ER等），新增该月汇总值
相对"滚动3年基线同口径分布"的分位数列（与 market_trend_volatility_check.py 第4.1节
同一套 rolling_window_distribution/percentile_of_recent 方法），避免只看绝对值无法
判断"这个月在历史上算不算极端"的问题——单位统一成分位数后才能和4.1节的结论直接对比。
分位数计算需要该月之前至少 hist_years 年的历史数据，故默认拉取窗口扩大到约4年
（3年基线 + 1年逐月展示窗口），--months 可指定展示的月数（不是拉取范围）。

ADX和ER分位数不是互相验证关系：ADX只看"方向是否反转"（净方向占优程度），
ER看"路径走得直不直"（净位移/总路径长度）。同方向但折返剧烈（ADX高、ER低）
和真正的大趋势（ADX高、ER高）在ADX上读数相近，必须结合ER才能区分，
单独用ADX判断趋势性会漏掉"走势是否迂回"这个对趋势策略止损消耗更直接的维度。

用法：
  python market_trend_volatility_monthly.py                       # 默认1d,8h,4h,1h，最近36个月
  python market_trend_volatility_monthly.py --timeframes 1d,4h,1h --months 12
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from trend_persistence_signal import (
    adx, efficiency_ratio, garman_klass_variance_series,
    rolling_window_distribution, percentile_of_recent,
    ASSETS, SYMBOL_MAP, HIST_WINDOW_DAYS,
)
from market_trend_volatility_check import fetch_ohlcv, atr_pct, BARS_PER_DAY

ANNUALIZE_DAYS = 365
# 逐月展示默认覆盖3年（与分位数基线窗口一致，第一年只用于给第二、三年的分位数计算
# 提供历史基线，本身不展示——分位数需要"该月之前3年"的数据，故实际拉取范围更长）
DEFAULT_MONTHS = 36
# 每种框架需要拉取的总天数：展示窗口 + 3年分位数基线 + 滑动分布回溯余量
FETCH_DAYS_FOR_MONTHLY = {
    "1w": DEFAULT_MONTHS * 31 + HIST_WINDOW_DAYS,
    "1d": DEFAULT_MONTHS * 31 + HIST_WINDOW_DAYS,
    "8h": DEFAULT_MONTHS * 31 + HIST_WINDOW_DAYS,
    "4h": DEFAULT_MONTHS * 31 + HIST_WINDOW_DAYS,
    "1h": DEFAULT_MONTHS * 31 + HIST_WINDOW_DAYS,
}


def monthly_table(df: pd.DataFrame, bars_per_day: float, months: int) -> pd.DataFrame:
    close = df["close"]
    gk_var = garman_klass_variance_series(df)  # 2026-11-09：波动率估计量从收盘价std换成Garman-Klass
    adx_n = max(int(round(14 * bars_per_day)), 3)
    er_n = max(int(round(30 * bars_per_day)), 5)
    hist_window_n = max(int(round(HIST_WINDOW_DAYS * bars_per_day)), adx_n)

    adx_series = adx(df, adx_n)
    er_series = efficiency_ratio(close, er_n)
    atr_series = atr_pct(df, adx_n)
    periods_per_year = bars_per_day * ANNUALIZE_DAYS

    frame = pd.DataFrame({
        "gk_var": gk_var,
        "adx": adx_series,
        "er": er_series,
        "atr_pct": atr_series,
    })
    frame["ym"] = frame.index.to_period("M")

    grouped = frame.groupby("ym").agg(
        vol_annualized=("gk_var", lambda s: np.sqrt(s.mean() * periods_per_year) * 100),
        adx=("adx", "mean"),
        er=("er", "mean"),
        atr_pct=("atr_pct", "mean"),
        n_bars=("gk_var", "count"),
    )
    # 已知bug（2026-11-09发现未修复）：无条件丢弃最后一个自然月，不判断该月是否只有部分数据
    # 当月天数不足时会被误当作"完整月"丢弃，曾误伤已有9天数据的11月，使用时注意月末数据可能缺失
    table = grouped.tail(months + 1).iloc[:-1] if len(grouped) > months else grouped.iloc[:-1]

    # 每个展示月的分位数：用该月之前hist_window_n根K线构造同口径分布。
    # 历史不足时（序列早期月份）分位数列留空，不是0或0%分位，避免混淆"无法判断"和"确实极端"。
    def vol_agg(window: np.ndarray) -> float:
        return np.sqrt(np.mean(window) * periods_per_year) * 100

    vol_pcts, adx_pcts, er_pcts = [], [], []
    for ym in table.index:
        month_end_idx = frame.index.get_indexer([ym.end_time], method="pad")[0]
        month_start_idx = frame.index.get_indexer([ym.start_time], method="backfill")[0]
        hist_start = max(0, month_start_idx - hist_window_n)
        if hist_start >= month_start_idx or month_start_idx - hist_start < hist_window_n:
            vol_pcts.append(None)
            adx_pcts.append(None)
            er_pcts.append(None)
            continue
        hist_slice = slice(hist_start, month_start_idx)
        window_n = month_end_idx - month_start_idx + 1
        vol_dist = rolling_window_distribution(gk_var.iloc[hist_slice], window_n, vol_agg)
        adx_dist = rolling_window_distribution(adx_series.iloc[hist_slice], window_n, np.mean)
        er_dist = rolling_window_distribution(er_series.iloc[hist_slice], window_n, np.mean)
        vol_pcts.append(percentile_of_recent(table.loc[ym, "vol_annualized"], vol_dist))
        adx_pcts.append(percentile_of_recent(table.loc[ym, "adx"], adx_dist))
        er_pcts.append(percentile_of_recent(table.loc[ym, "er"], er_dist))
    table["vol_percentile"] = vol_pcts
    table["adx_percentile"] = adx_pcts
    table["er_percentile"] = er_pcts
    return table


def run_timeframe(timeframe: str, months: int) -> None:
    bars_per_day = BARS_PER_DAY[timeframe]
    max_days = max(FETCH_DAYS_FOR_MONTHLY[timeframe], int((months + 2) * 31 + HIST_WINDOW_DAYS))
    print(f"\n{'='*70}\n时间框架：{timeframe}　最近{months}个月逐月走势（含相对3年基线的分位数）\n{'='*70}")

    tables = {}
    for asset in ASSETS:
        df = fetch_ohlcv(SYMBOL_MAP[asset], timeframe, max_days)
        tables[asset] = monthly_table(df, bars_per_day, months)

    def pct_str(p):
        return f"{p:.0f}%" if p is not None else "样本不足"

    for asset, t in tables.items():
        print(f"\n--- {asset} ({timeframe}) ---")
        print(f"{'月份':<10} {'年化波动率%':>10} {'波动率分位':>8} {'ADX':>8} {'ADX分位':>8} {'ER':>8} {'ER分位':>8} {'ATR%':>8} {'K线数':>6}")
        for ym, row in t.iterrows():
            print(f"{str(ym):<10} {row['vol_annualized']:>10.1f} {pct_str(row['vol_percentile']):>8} "
                  f"{row['adx']:>8.1f} {pct_str(row['adx_percentile']):>8} "
                  f"{row['er']:>8.3f} {pct_str(row['er_percentile']):>8} {row['atr_pct']:>8.2f} {row['n_bars']:>6.0f}")

    print(f"\n--- {timeframe} 跨标的月度均值 ---")
    all_months = sorted(set().union(*[set(t.index) for t in tables.values()]))
    print(f"{'月份':<10} {'年化波动率%':>10} {'波动率分位':>8} {'ADX':>8} {'ADX分位':>8} {'ER':>8} {'ER分位':>8} {'ATR%':>8}")
    for ym in all_months:
        vols = [tables[a].loc[ym, "vol_annualized"] for a in ASSETS if ym in tables[a].index]
        vol_ps = [tables[a].loc[ym, "vol_percentile"] for a in ASSETS if ym in tables[a].index and tables[a].loc[ym, "vol_percentile"] is not None]
        adxs = [tables[a].loc[ym, "adx"] for a in ASSETS if ym in tables[a].index]
        adx_ps = [tables[a].loc[ym, "adx_percentile"] for a in ASSETS if ym in tables[a].index and tables[a].loc[ym, "adx_percentile"] is not None]
        ers = [tables[a].loc[ym, "er"] for a in ASSETS if ym in tables[a].index]
        er_ps = [tables[a].loc[ym, "er_percentile"] for a in ASSETS if ym in tables[a].index and tables[a].loc[ym, "er_percentile"] is not None]
        atrs = [tables[a].loc[ym, "atr_pct"] for a in ASSETS if ym in tables[a].index]
        vol_p_str = f"{np.mean(vol_ps):.0f}%" if vol_ps else "样本不足"
        adx_p_str = f"{np.mean(adx_ps):.0f}%" if adx_ps else "样本不足"
        er_p_str = f"{np.mean(er_ps):.0f}%" if er_ps else "样本不足"
        print(f"{str(ym):<10} {np.mean(vols):>10.1f} {vol_p_str:>8} {np.mean(adxs):>8.1f} {adx_p_str:>8} "
              f"{np.mean(ers):>8.3f} {er_p_str:>8} {np.mean(atrs):>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="市场趋势性/波动性逐月走势")
    parser.add_argument("--timeframes", default="1d,8h,4h,1h", help="逗号分隔时间框架（默认1d,8h,4h,1h）")
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS, help=f"最近N个月（默认{DEFAULT_MONTHS}）")
    args = parser.parse_args()

    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf not in BARS_PER_DAY:
            print(f"不支持的时间框架：{tf}，可选：{list(BARS_PER_DAY)}")
            continue
        run_timeframe(tf, args.months)


if __name__ == "__main__":
    main()
