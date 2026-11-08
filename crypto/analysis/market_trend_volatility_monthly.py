"""
加密货币市场趋势性/波动性逐月走势（最近N个月）

背景：2026-11-04 market_trend_volatility_check.py 用"近期vs历史"二分对比确认了
波动率收缩跨周期一致、趋势性仅中短周期(4h/1h)显著减弱。本脚本把近期窗口按自然月
拆开，看这两个现象是从哪个月开始、是否逐月加剧，而不是只看一个笼统的"近N天"均值。

默认跑1d（基线参照）和4h（策略敏感粒度，此前发现趋势性减弱在此框架最明显）两档。
复用 market_trend_volatility_check.py 的拉取/指标函数。

用法：
  python market_trend_volatility_monthly.py                       # 默认1d,4h，最近6个月
  python market_trend_volatility_monthly.py --timeframes 1d,4h,1h --months 6
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from trend_persistence_signal import adx, efficiency_ratio, ASSETS, SYMBOL_MAP
from market_trend_volatility_check import fetch_ohlcv, atr_pct, BARS_PER_DAY, MAX_FETCH_DAYS

ANNUALIZE_DAYS = 365


def monthly_table(df: pd.DataFrame, bars_per_day: float, months: int) -> pd.DataFrame:
    close = df["close"]
    log_ret = np.log(close).diff()
    adx_n = max(int(round(14 * bars_per_day)), 3)
    er_n = max(int(round(30 * bars_per_day)), 5)

    adx_series = adx(df, adx_n)
    er_series = efficiency_ratio(close, er_n)
    atr_series = atr_pct(df, adx_n)
    periods_per_year = bars_per_day * ANNUALIZE_DAYS

    frame = pd.DataFrame({
        "log_ret": log_ret,
        "adx": adx_series,
        "er": er_series,
        "atr_pct": atr_series,
    })
    frame["ym"] = frame.index.to_period("M")

    grouped = frame.groupby("ym").agg(
        vol_annualized=("log_ret", lambda s: s.std() * np.sqrt(periods_per_year) * 100),
        adx=("adx", "mean"),
        er=("er", "mean"),
        atr_pct=("atr_pct", "mean"),
        n_bars=("log_ret", "count"),
    )
    return grouped.tail(months + 1).iloc[:-1] if len(grouped) > months else grouped.iloc[:-1]


def run_timeframe(timeframe: str, months: int) -> None:
    bars_per_day = BARS_PER_DAY[timeframe]
    max_days = max(MAX_FETCH_DAYS[timeframe], (months + 2) * 31)
    print(f"\n{'='*70}\n时间框架：{timeframe}　最近{months}个月逐月走势\n{'='*70}")

    tables = {}
    for asset in ASSETS:
        df = fetch_ohlcv(SYMBOL_MAP[asset], timeframe, max_days)
        tables[asset] = monthly_table(df, bars_per_day, months)

    for asset, t in tables.items():
        print(f"\n--- {asset} ({timeframe}) ---")
        print(f"{'月份':<10} {'年化波动率%':>10} {'ADX':>8} {'ER':>8} {'ATR%':>8} {'K线数':>6}")
        for ym, row in t.iterrows():
            print(f"{str(ym):<10} {row['vol_annualized']:>10.1f} {row['adx']:>8.1f} "
                  f"{row['er']:>8.3f} {row['atr_pct']:>8.2f} {row['n_bars']:>6.0f}")

    print(f"\n--- {timeframe} 跨标的月度均值 ---")
    all_months = sorted(set().union(*[set(t.index) for t in tables.values()]))
    print(f"{'月份':<10} {'年化波动率%':>10} {'ADX':>8} {'ER':>8} {'ATR%':>8}")
    for ym in all_months:
        vols = [tables[a].loc[ym, "vol_annualized"] for a in ASSETS if ym in tables[a].index]
        adxs = [tables[a].loc[ym, "adx"] for a in ASSETS if ym in tables[a].index]
        ers = [tables[a].loc[ym, "er"] for a in ASSETS if ym in tables[a].index]
        atrs = [tables[a].loc[ym, "atr_pct"] for a in ASSETS if ym in tables[a].index]
        print(f"{str(ym):<10} {np.mean(vols):>10.1f} {np.mean(adxs):>8.1f} "
              f"{np.mean(ers):>8.3f} {np.mean(atrs):>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="市场趋势性/波动性逐月走势")
    parser.add_argument("--timeframes", default="1d,4h", help="逗号分隔时间框架（默认1d,4h）")
    parser.add_argument("--months", type=int, default=6, help="最近N个月（默认6）")
    args = parser.parse_args()

    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf not in BARS_PER_DAY:
            print(f"不支持的时间框架：{tf}，可选：{list(BARS_PER_DAY)}")
            continue
        run_timeframe(tf, args.months)


if __name__ == "__main__":
    main()
