"""
加密货币市场趋势性/波动性近期 vs 历史基线对比（多时间框架）

背景：2026-11-04 attribution_check.py归因发现4个策略MFE（趋势耗尽）信号一致走弱，
怀疑是市场整体趋势性/波动性系统性减弱，而非策略归因层面的倒推。本脚本独立于策略
PnL，直接用价格序列测量市场本身的趋势强度和波动率，近期窗口 vs 历史基线对比。

支持多时间框架（1w/1d/8h/4h/1h）：策略实际跑在205m/479m/3h等中短周期上，
只看日线会被"开高低收"压缩抹掉盘中来回震荡的幅度，可能漏掉中短周期的变化。
周期参数（ADX/ATR窗口、近期/历史窗口）按"自然天数"换算成对应时间框架的K线数，
确保跨框架可比（如14天在1d上是14根，在4h上是84根）。

复用 trend_persistence_signal.py 的 adx/hurst_exponent/efficiency_ratio。

用法：
  python market_trend_volatility_check.py                                  # 默认1d，近90天 vs 历史全周期
  python market_trend_volatility_check.py --timeframes 1w,1d,8h,4h,1h      # 多框架对比
  python market_trend_volatility_check.py --timeframes 4h --recent-days 30 --hist-days 180
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import ccxt
import numpy as np
import pandas as pd

from trend_persistence_signal import adx, hurst_exponent, efficiency_ratio, ASSETS, SYMBOL_MAP

# 每种时间框架一天对应的K线数，用于把"天数"参数换算成K线数
BARS_PER_DAY = {"1w": 1 / 7, "1d": 1, "8h": 3, "4h": 6, "1h": 24}
# 短周期K线数据量大，不需要拉全历史——按时间框架限制最多拉取天数
# 1h 的 ADX(14天=336根)在EWM平滑下需要约5倍窗口才收敛，若历史基线太短会被初始值拖累失真，
# 故 1h 拉取天数需显著大于 recent_days + adx窗口对应天数的5倍（约30+70=100天不够，提到400天）
MAX_FETCH_DAYS = {"1w": 3000, "1d": 3000, "8h": 720, "4h": 400, "1h": 400}
ANNUALIZE_DAYS = 365


def fetch_ohlcv(symbol: str, timeframe: str, max_days: int) -> pd.DataFrame:
    print(f"  拉取 {symbol} {timeframe}...", end=" ", flush=True)
    ex = ccxt.binance({"options": {"defaultType": "future"}, "timeout": 30000})
    since = ex.milliseconds() - max_days * 24 * 3600 * 1000
    all_bars = []
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < 1000:
            break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("dt").sort_index().drop_duplicates()
    print(f"done ({len(df)} bars)")
    return df


def atr_pct(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr / close * 100


def summarize(df: pd.DataFrame, bars_per_day: float, recent_days: int, hist_days: int | None) -> dict:
    """recent_days/hist_days 均按自然天数指定，内部换算成K线数。hist_days=None 表示用剩余全部历史。"""
    recent_n = max(int(round(recent_days * bars_per_day)), 5)
    adx_n = max(int(round(14 * bars_per_day)), 3)
    er_n = max(int(round(30 * bars_per_day)), 5)

    close = df["close"]
    log_ret = np.log(close).diff().dropna()
    adx_series = adx(df, adx_n)
    er_series = efficiency_ratio(close, er_n)
    atr_series = atr_pct(df, adx_n)

    if hist_days is not None:
        hist_n = max(int(round(hist_days * bars_per_day)), recent_n + adx_n)
        base = log_ret.iloc[-(recent_n + hist_n):-recent_n] if recent_n + hist_n <= len(log_ret) else log_ret.iloc[:-recent_n]
        hist_slice = slice(-(recent_n + hist_n), -recent_n)
    else:
        hist_slice = slice(0, -recent_n)

    recent_log_ret = log_ret.iloc[-recent_n:]
    recent_adx = adx_series.iloc[-recent_n:]
    recent_er = er_series.iloc[-recent_n:]
    recent_atr = atr_series.iloc[-recent_n:]
    recent_hurst = hurst_exponent(recent_log_ret.values)

    hist_log_ret = log_ret.iloc[hist_slice]
    hist_adx = adx_series.iloc[hist_slice]
    hist_er = er_series.iloc[hist_slice]
    hist_atr = atr_series.iloc[hist_slice]

    periods_per_year = bars_per_day * ANNUALIZE_DAYS
    return {
        "recent_vol_annualized": recent_log_ret.std() * np.sqrt(periods_per_year) * 100,
        "hist_vol_annualized": hist_log_ret.std() * np.sqrt(periods_per_year) * 100,
        "recent_atr_pct": recent_atr.mean(),
        "hist_atr_pct": hist_atr.mean(),
        "recent_adx": recent_adx.mean(),
        "hist_adx": hist_adx.mean(),
        "recent_er": recent_er.mean(),
        "hist_er": hist_er.mean(),
        "recent_hurst": recent_hurst,
        "n_recent_bars": recent_n,
        "n_hist_bars": len(hist_log_ret),
    }


def run_timeframe(timeframe: str, recent_days: int, hist_days: int | None) -> None:
    bars_per_day = BARS_PER_DAY[timeframe]
    max_days = MAX_FETCH_DAYS[timeframe]
    print(f"\n{'='*60}\n时间框架：{timeframe}（近{recent_days}天 vs "
          f"{'历史全周期' if hist_days is None else f'历史前{hist_days}天'}）\n{'='*60}\n")

    rows = []
    for asset in ASSETS:
        df = fetch_ohlcv(SYMBOL_MAP[asset], timeframe, max_days)
        s = summarize(df, bars_per_day, recent_days, hist_days)
        rows.append((asset, s))

        print(f"--- {asset} ---  (近期{s['n_recent_bars']}根 / 历史{s['n_hist_bars']}根)")
        print(f"  年化波动率：近期{s['recent_vol_annualized']:.1f}% vs 历史{s['hist_vol_annualized']:.1f}%  "
              f"(比值{s['recent_vol_annualized']/s['hist_vol_annualized']:.2f})")
        print(f"  ATR%：近期{s['recent_atr_pct']:.2f}% vs 历史{s['hist_atr_pct']:.2f}%  "
              f"(比值{s['recent_atr_pct']/s['hist_atr_pct']:.2f})")
        print(f"  ADX：近期{s['recent_adx']:.1f} vs 历史{s['hist_adx']:.1f}  "
              f"(比值{s['recent_adx']/s['hist_adx']:.2f})")
        print(f"  效率比ER：近期{s['recent_er']:.3f} vs 历史{s['hist_er']:.3f}  "
              f"(比值{s['recent_er']/s['hist_er']:.2f})")
        print()

    print(f"----- {timeframe} 跨标的汇总 -----")
    vol_ratios = [s["recent_vol_annualized"] / s["hist_vol_annualized"] for _, s in rows]
    adx_ratios = [s["recent_adx"] / s["hist_adx"] for _, s in rows]
    er_ratios = [s["recent_er"] / s["hist_er"] for _, s in rows]
    print(f"  波动率比值：{dict(zip(ASSETS, [round(r,2) for r in vol_ratios]))}")
    print(f"  ADX比值：{dict(zip(ASSETS, [round(r,2) for r in adx_ratios]))}")
    print(f"  效率比比值：{dict(zip(ASSETS, [round(r,2) for r in er_ratios]))}")
    n_below_adx = sum(1 for r in adx_ratios if r < 0.85)
    n_below_vol = sum(1 for r in vol_ratios if r < 0.85)
    print(f"  ADX<0.85（趋势显著减弱）：{n_below_adx}/{len(ASSETS)}  "
          f"波动率<0.85（波动显著减弱）：{n_below_vol}/{len(ASSETS)}")


def main():
    parser = argparse.ArgumentParser(description="市场趋势性/波动性近期vs历史基线对比，支持多时间框架")
    parser.add_argument("--timeframes", default="1d",
                        help="逗号分隔的时间框架，如 1w,1d,8h,4h,1h（默认1d）")
    parser.add_argument("--recent-days", type=int, default=90,
                        help="近期窗口天数（默认90，按自然天数换算成各框架的K线数）")
    parser.add_argument("--hist-days", type=int, default=None,
                        help="历史基线窗口天数（默认None=用剩余全部历史）")
    args = parser.parse_args()

    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf not in BARS_PER_DAY:
            print(f"不支持的时间框架：{tf}，可选：{list(BARS_PER_DAY)}")
            continue
        run_timeframe(tf, args.recent_days, args.hist_days)


if __name__ == "__main__":
    main()
