"""
趋势段持续长度 / 方向反转频率：检验"趋势反复打断"假设

背景：2026-11-04 调研确认市场波动率收缩、趋势性（ADX/ER）仅中短周期减弱，但这只衡量
"趋势强度均值"，没有回答策略受损的真实机制是"方向性持续缺失"还是"趋势反复中途被打断"
（高频进出、止损消耗）。本脚本直接测量两个新指标来区分这两种机制：

  1. 趋势段平均长度：ADX上穿25记为趋势段开始，下穿20记为结束（标准技术分析口径，
     上下穿用不同阈值形成滞后带，避免ADX在25附近噪音反复触发产生大量假段）。
     最近90天 vs 历史3年滚动基线比较平均段长（按bar数）。分位数<25视为
     "趋势段显著变短"。

  2. 方向反转频率：用20日EMA斜率符号作为方向代理（周期按天数换算成K线数，跨框架
     可比），统计斜率符号翻转次数，换算成"每100根K线翻转次数"。最近90天 vs 历史
     3年滚动基线对比，分位数>75视为"反转显著更频繁"。

如果两个分位数确认（变短/更频繁），说明"趋势反复打断"假设成立；如果接近中位数，
说明这个新机制假设不成立，受损机制另有原因（如纯粹的趋势强度/方向性缺失）。

2026-11-09方法论改进：历史基线从"剩余全部历史"改为"滚动3年"，比例阈值（0.85/1.15）
改为分位数阈值（PCT_LOW/PCT_HIGH）。注：历史分布由"3年窗口内按ROLLING_STEP步长切出
与recent_n等长的子窗口，每个子窗口各自重跑一次段检测/翻转统计"构造——段长度、翻转
次数不是逐bar可加总的量，不能直接套用trend_persistence_signal.rolling_window_
distribution（那是对逐bar序列做滑动均值，这里的agg本身就是一次完整的段检测算法）。
已知限制：该指标在90天窗口下样本量本就小（历史上多数90天子窗口段数0-1），3年滚动
分布会出现大量子窗口mean_len=NaN，分位数计算显式过滤NaN，有效样本<10个时判定为
"样本不足"而非输出一个不可信的数字。

复用 trend_persistence_signal.py 的 adx/ASSETS/SYMBOL_MAP/HIST_WINDOW_DAYS/PCT_LOW/
PCT_HIGH/ROLLING_STEP/percentile_of_recent，market_trend_volatility_check.py 的
fetch_ohlcv/BARS_PER_DAY/MAX_FETCH_DAYS。

用法：
  python trend_segment_length.py                          # 默认1d,4h，近90天 vs 历史3年滚动基线
  python trend_segment_length.py --timeframes 1d,4h,1h --recent-days 90
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
    adx, ASSETS, SYMBOL_MAP, HIST_WINDOW_DAYS, PCT_LOW, PCT_HIGH,
    ROLLING_STEP, percentile_of_recent,
)
from market_trend_volatility_check import fetch_ohlcv, BARS_PER_DAY, MAX_FETCH_DAYS

ADX_ENTER = 25  # 趋势段开始阈值（上穿）
ADX_EXIT = 20   # 趋势段结束阈值（下穿），低于进入阈值形成滞后带防抖
EMA_DAYS = 20   # 方向代理用的EMA周期（自然天数，换算成各框架K线数）


def find_trend_segments(adx_series: pd.Series) -> list[int]:
    """返回每个趋势段的长度（bar数）。段定义：ADX上穿ADX_ENTER开始，下穿ADX_EXIT结束。"""
    vals = adx_series.values
    segments = []
    in_segment = False
    start_idx = None
    for i in range(1, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            continue
        if not in_segment and vals[i - 1] <= ADX_ENTER < vals[i]:
            in_segment = True
            start_idx = i
        elif in_segment and vals[i - 1] >= ADX_EXIT > vals[i]:
            segments.append(i - start_idx)
            in_segment = False
    return segments


def count_slope_flips(close: pd.Series, ema_n: int) -> pd.Series:
    """返回EMA斜率符号序列（+1/-1/0），用于统计翻转次数。"""
    ema = close.ewm(span=ema_n, adjust=False).mean()
    slope = ema.diff()
    return np.sign(slope)


def segment_stats(segments: list[int]) -> dict:
    if not segments:
        return {"n_segments": 0, "mean_len": np.nan}
    return {"n_segments": len(segments), "mean_len": float(np.mean(segments))}


def flip_rate_per_100(sign_series: pd.Series) -> float:
    s = sign_series[sign_series != 0]
    if len(s) < 2:
        return np.nan
    flips = (np.diff(s.values) != 0).sum()
    return flips / len(s) * 100


def _sub_window_distribution(adx_series: pd.Series, slope_sign: pd.Series, window_n: int, step: int) -> tuple:
    """在历史区间内，按step步长切出与window_n等长的子窗口，每个子窗口各自重跑一次
    段检测（mean_len）和翻转统计（flip_rate），返回两个分布数组（已过滤NaN）。
    """
    n = len(adx_series)
    seg_dist, flip_dist = [], []
    for start in range(0, n - window_n + 1, step):
        end = start + window_n
        sub_adx = adx_series.iloc[start:end]
        sub_segments = find_trend_segments(sub_adx)
        sub_seg_stats = segment_stats(sub_segments)
        if not np.isnan(sub_seg_stats["mean_len"]):
            seg_dist.append(sub_seg_stats["mean_len"])

        sub_slope = slope_sign.iloc[start:end].dropna()
        sub_flip_rate = flip_rate_per_100(sub_slope)
        if not np.isnan(sub_flip_rate):
            flip_dist.append(sub_flip_rate)
    return np.array(seg_dist), np.array(flip_dist)


def analyze(df: pd.DataFrame, bars_per_day: float, recent_days: int) -> dict:
    adx_n = max(int(round(14 * bars_per_day)), 3)
    ema_n = max(int(round(EMA_DAYS * bars_per_day)), 3)
    recent_n = max(int(round(recent_days * bars_per_day)), 10)
    hist_window_n = max(int(round(HIST_WINDOW_DAYS * bars_per_day)), recent_n + adx_n)

    adx_series = adx(df, adx_n)
    close = df["close"]

    available_hist_n = len(df) - recent_n
    hist_insufficient = available_hist_n < hist_window_n
    use_hist_n = min(hist_window_n, available_hist_n)
    hist_slice = slice(-(recent_n + use_hist_n), -recent_n)

    recent_adx = adx_series.iloc[-recent_n:]
    hist_adx = adx_series.iloc[hist_slice]

    recent_segments = find_trend_segments(recent_adx)
    hist_segments = find_trend_segments(hist_adx)
    recent_seg_stats = segment_stats(recent_segments)
    hist_seg_stats = segment_stats(hist_segments)

    slope_sign = count_slope_flips(close, ema_n)
    recent_slope = slope_sign.iloc[-recent_n:].dropna()
    hist_slope = slope_sign.iloc[hist_slice].dropna()

    recent_flip_rate = flip_rate_per_100(recent_slope)
    hist_flip_rate = flip_rate_per_100(hist_slope)

    seg_dist, flip_dist = _sub_window_distribution(hist_adx, slope_sign.iloc[hist_slice], recent_n, ROLLING_STEP)

    return {
        "n_recent_bars": recent_n,
        "n_hist_bars": use_hist_n,
        "hist_insufficient": hist_insufficient,
        "recent_n_segments": recent_seg_stats["n_segments"],
        "recent_mean_seg_len": recent_seg_stats["mean_len"],
        "hist_n_segments": hist_seg_stats["n_segments"],
        "hist_mean_seg_len": hist_seg_stats["mean_len"],
        "seg_len_ratio": (recent_seg_stats["mean_len"] / hist_seg_stats["mean_len"]
                           if hist_seg_stats["mean_len"] else np.nan),
        "seg_len_percentile": percentile_of_recent(recent_seg_stats["mean_len"], seg_dist),
        "recent_flip_rate": recent_flip_rate,
        "hist_flip_rate": hist_flip_rate,
        "flip_rate_ratio": (recent_flip_rate / hist_flip_rate
                             if hist_flip_rate not in (0, np.nan) else np.nan),
        "flip_rate_percentile": percentile_of_recent(recent_flip_rate, flip_dist),
    }


def _pct_str(p: float | None) -> str:
    return f"{p:.0f}%分位" if p is not None else "样本不足"


def run_timeframe(timeframe: str, recent_days: int) -> None:
    bars_per_day = BARS_PER_DAY[timeframe]
    max_days = MAX_FETCH_DAYS[timeframe]
    print(f"\n{'='*70}\n时间框架：{timeframe}　趋势段长度/反转频率（近{recent_days}天 vs 历史3年滚动基线）\n{'='*70}\n")

    rows = []
    for asset in ASSETS:
        df = fetch_ohlcv(SYMBOL_MAP[asset], timeframe, max_days)
        s = analyze(df, bars_per_day, recent_days)
        rows.append((asset, s))

        insuff_tag = "　⚠️基线不足3年，分位数置信度降低" if s["hist_insufficient"] else ""
        print(f"--- {asset} ({timeframe}) --- (近期{s['n_recent_bars']}根 / 历史{s['n_hist_bars']}根){insuff_tag}")
        print(f"  趋势段数：近期{s['recent_n_segments']}段 / 历史{s['hist_n_segments']}段")
        print(f"  趋势段平均长度(bar)：近期{s['recent_mean_seg_len']:.1f} vs 历史{s['hist_mean_seg_len']:.1f}  "
              f"({_pct_str(s['seg_len_percentile'])})")
        print(f"  方向反转频率(次/100bar)：近期{s['recent_flip_rate']:.2f} vs 历史{s['hist_flip_rate']:.2f}  "
              f"({_pct_str(s['flip_rate_percentile'])})")
        print()

    print(f"----- {timeframe} 跨标的汇总 -----")
    seg_pcts = {a: s["seg_len_percentile"] for a, s in rows}
    flip_pcts = {a: s["flip_rate_percentile"] for a, s in rows}
    print(f"  趋势段长度分位数：{ {a: (round(p, 0) if p is not None else None) for a, p in seg_pcts.items()} }")
    print(f"  反转频率分位数：{ {a: (round(p, 0) if p is not None else None) for a, p in flip_pcts.items()} }")
    n_shorter = sum(1 for p in seg_pcts.values() if p is not None and p < PCT_LOW)
    n_more_freq = sum(1 for p in flip_pcts.values() if p is not None and p > PCT_HIGH)
    print(f"  趋势段<{PCT_LOW}分位（显著变短）：{n_shorter}/{len(ASSETS)}  "
          f"反转频率>{PCT_HIGH}分位（显著更频繁）：{n_more_freq}/{len(ASSETS)}")


def main():
    parser = argparse.ArgumentParser(description="趋势段持续长度/方向反转频率：检验趋势反复打断假设")
    parser.add_argument("--timeframes", default="1d,4h", help="逗号分隔时间框架（默认1d,4h）")
    parser.add_argument("--recent-days", type=int, default=90, help="近期窗口天数（默认90）")
    args = parser.parse_args()

    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf not in BARS_PER_DAY:
            print(f"不支持的时间框架：{tf}，可选：{list(BARS_PER_DAY)}")
            continue
        run_timeframe(tf, args.recent_days)


if __name__ == "__main__":
    main()
