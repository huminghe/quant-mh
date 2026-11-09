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

2026-11-09方法论改进：波动率估计量从收盘价std换成Garman-Klass（用OHLC四价，
统计效率更高）；历史基线从"剩余全部历史均值"改为"滚动3年+分位数"，避免被
2020-2021年极端行情拉高基线产生系统性bias；"比值<0.85"判断换成"历史3年滚动
分布的分位数<25"，不再借用未经校验的盈亏比监控阈值。

用法：
  python market_trend_volatility_check.py                                  # 默认1d，近90天 vs 历史3年
  python market_trend_volatility_check.py --timeframes 1w,1d,8h,4h,1h      # 多框架对比
  python market_trend_volatility_check.py --timeframes 4h --recent-days 30 --hist-years 3
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

from trend_persistence_signal import (
    adx, hurst_exponent, efficiency_ratio, garman_klass_variance_series,
    rolling_window_distribution, percentile_of_recent,
    ASSETS, SYMBOL_MAP, PCT_LOW, PCT_HIGH,
)

# 每种时间框架一天对应的K线数，用于把"天数"参数换算成K线数
BARS_PER_DAY = {"1w": 1 / 7, "1d": 1, "8h": 3, "4h": 6, "1h": 24}
# 短周期K线数据量大，不需要拉全历史——按时间框架限制最多拉取天数
# 1h 的 ADX(14天=336根)在EWM平滑下需要约5倍窗口才收敛，若历史基线太短会被初始值拖累失真，
# 故 1h 拉取天数需显著大于 recent_days + adx窗口对应天数的5倍（约30+70=100天不够，提到400天）
# 2026-11-09：历史基线固定3年（约1095天），各框架拉取天数需覆盖"3年基线+滑动分布回溯余量"，
# 故全部提高到约1500天（1d/1w从3000天降到1500天，同样够用，减少不必要拉取）
MAX_FETCH_DAYS = {"1w": 1500, "1d": 1500, "8h": 1500, "4h": 1500, "1h": 1500}
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


def summarize(df: pd.DataFrame, bars_per_day: float, recent_days: int, hist_years: float = 3.0) -> dict:
    """recent_days 按自然天数指定，内部换算成K线数。历史基线固定为 hist_years 年滚动窗口
    （2026-11-09起不再支持"剩余全部历史"，避免被极端行情年份拉高基线产生系统性bias）。
    若可用历史不足该窗口，用"有多少用多少"，并在返回dict标记 hist_insufficient=True。
    """
    recent_n = max(int(round(recent_days * bars_per_day)), 5)
    adx_n = max(int(round(14 * bars_per_day)), 3)
    er_n = max(int(round(30 * bars_per_day)), 5)
    hist_window_n = max(int(round(hist_years * 365 * bars_per_day)), recent_n + adx_n)

    close = df["close"]
    log_ret = np.log(close).diff().dropna()
    gk_var = garman_klass_variance_series(df)
    adx_series = adx(df, adx_n)
    er_series = efficiency_ratio(close, er_n)
    atr_series = atr_pct(df, adx_n)

    available_hist_n = len(df) - recent_n
    hist_insufficient = available_hist_n < hist_window_n
    use_hist_n = min(hist_window_n, available_hist_n)
    hist_slice = slice(-(recent_n + use_hist_n), -recent_n)

    periods_per_year = bars_per_day * ANNUALIZE_DAYS

    recent_gk_var = gk_var.iloc[-recent_n:]
    recent_adx = adx_series.iloc[-recent_n:]
    recent_er = er_series.iloc[-recent_n:]
    recent_atr = atr_series.iloc[-recent_n:]
    recent_hurst = hurst_exponent(log_ret.iloc[-recent_n:].values)
    recent_vol = np.sqrt(recent_gk_var.mean() * periods_per_year) * 100

    hist_gk_var = gk_var.iloc[hist_slice]
    hist_adx = adx_series.iloc[hist_slice]
    hist_er = er_series.iloc[hist_slice]
    hist_atr = atr_series.iloc[hist_slice]
    hist_vol = np.sqrt(hist_gk_var.mean() * periods_per_year) * 100

    # 分位数：近期窗口汇总值在"3年历史滚动同口径分布"中的位置。
    # 子窗口含任何NaN（如序列开头warm-up期）即跳过，由rolling_window_distribution统一处理。
    def vol_agg(window: np.ndarray) -> float:
        return np.sqrt(np.mean(window) * periods_per_year) * 100

    vol_dist = rolling_window_distribution(gk_var.iloc[hist_slice], recent_n, vol_agg)
    adx_dist = rolling_window_distribution(adx_series.iloc[hist_slice], recent_n, np.mean)
    er_dist = rolling_window_distribution(er_series.iloc[hist_slice], recent_n, np.mean)
    atr_dist = rolling_window_distribution(atr_series.iloc[hist_slice], recent_n, np.mean)

    return {
        "recent_vol_annualized": recent_vol,
        "hist_vol_annualized": hist_vol,
        "vol_percentile": percentile_of_recent(recent_vol, vol_dist),
        "recent_atr_pct": recent_atr.mean(),
        "hist_atr_pct": hist_atr.mean(),
        "atr_percentile": percentile_of_recent(recent_atr.mean(), atr_dist),
        "recent_adx": recent_adx.mean(),
        "hist_adx": hist_adx.mean(),
        "adx_percentile": percentile_of_recent(recent_adx.mean(), adx_dist),
        "recent_er": recent_er.mean(),
        "hist_er": hist_er.mean(),
        "er_percentile": percentile_of_recent(recent_er.mean(), er_dist),
        "recent_hurst": recent_hurst,
        "n_recent_bars": recent_n,
        "n_hist_bars": use_hist_n,
        "hist_insufficient": hist_insufficient,
    }


def _pct_str(p: float | None) -> str:
    return f"{p:.0f}%分位" if p is not None else "样本不足"


def run_timeframe(timeframe: str, recent_days: int, hist_years: float) -> None:
    bars_per_day = BARS_PER_DAY[timeframe]
    max_days = MAX_FETCH_DAYS[timeframe]
    print(f"\n{'='*60}\n时间框架：{timeframe}（近{recent_days}天 vs 历史{hist_years}年滚动基线）\n{'='*60}\n")

    rows = []
    for asset in ASSETS:
        df = fetch_ohlcv(SYMBOL_MAP[asset], timeframe, max_days)
        s = summarize(df, bars_per_day, recent_days, hist_years)
        rows.append((asset, s))

        insuff_tag = "　⚠️基线不足3年，分位数置信度降低" if s["hist_insufficient"] else ""
        print(f"--- {asset} ---  (近期{s['n_recent_bars']}根 / 历史{s['n_hist_bars']}根){insuff_tag}")
        print(f"  年化波动率：近期{s['recent_vol_annualized']:.1f}% vs 历史{s['hist_vol_annualized']:.1f}%  "
              f"({_pct_str(s['vol_percentile'])})")
        print(f"  ATR%：近期{s['recent_atr_pct']:.2f}% vs 历史{s['hist_atr_pct']:.2f}%  "
              f"({_pct_str(s['atr_percentile'])})")
        print(f"  ADX：近期{s['recent_adx']:.1f} vs 历史{s['hist_adx']:.1f}  "
              f"({_pct_str(s['adx_percentile'])})")
        print(f"  效率比ER：近期{s['recent_er']:.3f} vs 历史{s['hist_er']:.3f}  "
              f"({_pct_str(s['er_percentile'])})")
        print()

    print(f"----- {timeframe} 跨标的汇总 -----")
    adx_pcts = [s["adx_percentile"] for _, s in rows]
    vol_pcts = [s["vol_percentile"] for _, s in rows]
    print(f"  ADX分位数：{dict(zip(ASSETS, [round(p, 0) if p is not None else None for p in adx_pcts]))}")
    print(f"  波动率分位数：{dict(zip(ASSETS, [round(p, 0) if p is not None else None for p in vol_pcts]))}")
    n_below_adx = sum(1 for p in adx_pcts if p is not None and p < PCT_LOW)
    n_below_vol = sum(1 for p in vol_pcts if p is not None and p < PCT_LOW)
    print(f"  ADX<{PCT_LOW}分位（趋势显著减弱）：{n_below_adx}/{len(ASSETS)}  "
          f"波动率<{PCT_LOW}分位（波动显著减弱）：{n_below_vol}/{len(ASSETS)}")


def main():
    parser = argparse.ArgumentParser(description="市场趋势性/波动性近期vs历史基线对比，支持多时间框架")
    parser.add_argument("--timeframes", default="1d",
                        help="逗号分隔的时间框架，如 1w,1d,8h,4h,1h（默认1d）")
    parser.add_argument("--recent-days", type=int, default=90,
                        help="近期窗口天数（默认90，按自然天数换算成各框架的K线数）")
    parser.add_argument("--hist-days", type=int, default=None,
                        help="[已废弃，2026-11-09起历史基线固定为--hist-years滚动窗口，本参数不再生效]")
    parser.add_argument("--hist-years", type=float, default=3.0,
                        help="历史基线滚动窗口年数（默认3.0，可传小数）")
    args = parser.parse_args()

    if args.hist_days is not None:
        print(f"⚠️ --hist-days已废弃（值{args.hist_days}被忽略），历史基线固定为--hist-years={args.hist_years}年滚动窗口")

    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf not in BARS_PER_DAY:
            print(f"不支持的时间框架：{tf}，可选：{list(BARS_PER_DAY)}")
            continue
        run_timeframe(tf, args.recent_days, args.hist_years)


if __name__ == "__main__":
    main()
