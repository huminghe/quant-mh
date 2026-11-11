"""
单bar尖峰（"门结构"）检测：候选新指标，比月度ADX/ER/GK均值更直观地反映
"大部分时间窄幅震荡、偶发单bar剧烈双向尖峰"这种现象。

背景：2026-11-11归因分析发现，9个部署策略约51%的近期亏损来自少数门结构
事件（如11-05 BTC/ETH/SOL/DOGE单bar跌5.5%-7%），但月度ADX/ER/GK方差均值
对这类事件天然钝化（被窗口内其他bar平滑掉），看不出发生过几次、多严重。
本脚本直接逐月统计"异常bar"数量，用两种判定方式：

1. ATR倍数法：|单bar收益| > atr_mult * ATR% ，直观、复用现有ATR基础设施
2. GK历史分位数法：GK方差 > 历史3年滚动分布的第 pct 分位，统计上更严谨

两种方式本质都是"固定/统计阈值识别异常bar"，用同一份数据算两遍，比较
哪种定义更稳定、误报率更低——不是两个独立指标，是同一候选的两种参数化。

用法：
  python spike_bar_detector.py --timeframe 4h --months 6
  python spike_bar_detector.py --timeframe 4h --months 6 --method gk_pct --pct 95
  python spike_bar_detector.py --timeframe 4h --months 6 --method atr_mult --atr-mult 3
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
    garman_klass_variance_series, rolling_window_distribution,
    HIST_WINDOW_DAYS, ASSETS, SYMBOL_MAP,
)
from market_trend_volatility_check import fetch_ohlcv, atr_pct, BARS_PER_DAY, MAX_FETCH_DAYS


def detect_spikes_atr_mult(df: pd.DataFrame, atr_n: int, atr_mult: float) -> pd.Series:
    """|单bar收益%| > atr_mult * ATR% 判定为尖峰bar，返回布尔Series。"""
    bar_ret_pct = df["close"].pct_change().abs() * 100
    atr_series = atr_pct(df, atr_n)
    return bar_ret_pct > (atr_mult * atr_series)


def detect_spikes_gk_pct(df: pd.DataFrame, hist_n: int, pct: float) -> pd.Series:
    """GK方差 > 历史hist_n根bar滚动分布的pct分位 判定为尖峰bar，返回布尔Series。
    用逐bar值本身（而非窗口汇总值）构造历史分布，因为这里要判定的是"单bar"级别
    的异常，不是窗口均值异常——与trend_persistence_signal.py里判定"窗口均值是否
    处于历史高/低位"是不同粒度的问题，不能直接复用rolling_window_distribution。
    """
    gk = garman_klass_variance_series(df)
    threshold = gk.rolling(hist_n, min_periods=max(30, hist_n // 10)).quantile(pct / 100)
    return gk > threshold


def monthly_spike_counts(df: pd.DataFrame, spike_mask: pd.Series) -> pd.DataFrame:
    """按月汇总：该月bar总数、尖峰bar数、尖峰占比。"""
    month = df.index.to_series().dt.to_period("M")
    total = month.groupby(month).count()
    spikes = month[spike_mask.values].groupby(month[spike_mask.values]).count()
    out = pd.DataFrame({"total_bars": total, "spike_bars": spikes.reindex(total.index).fillna(0).astype(int)})
    out["spike_pct"] = (out["spike_bars"] / out["total_bars"] * 100).round(1)
    return out


def main():
    ap = argparse.ArgumentParser(description="单bar尖峰（门结构）检测，逐月统计")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--months", type=int, default=6, help="展示最近几个月")
    ap.add_argument("--method", choices=["atr_mult", "gk_pct"], default="atr_mult")
    ap.add_argument("--atr-mult", type=float, default=3.0, help="atr_mult法：超过N倍ATR判定为尖峰")
    ap.add_argument("--atr-days", type=float, default=14, help="ATR窗口天数")
    ap.add_argument("--pct", type=float, default=95.0, help="gk_pct法：历史分位数阈值")
    args = ap.parse_args()

    tf = args.timeframe
    bars_per_day = BARS_PER_DAY[tf]
    atr_n = max(int(round(args.atr_days * bars_per_day)), 3)
    hist_n = max(int(round(HIST_WINDOW_DAYS * bars_per_day)), 100)

    print(f"方法: {args.method}  时间框架: {tf}")
    if args.method == "atr_mult":
        print(f"判定: |单bar收益%| > {args.atr_mult} × ATR({args.atr_days}天)")
    else:
        print(f"判定: GK方差 > 历史{HIST_WINDOW_DAYS}天滚动窗口第{args.pct}分位")
    print()

    for name in ASSETS:
        symbol = SYMBOL_MAP[name]
        df = fetch_ohlcv(symbol, tf, MAX_FETCH_DAYS[tf])

        if args.method == "atr_mult":
            spike_mask = detect_spikes_atr_mult(df, atr_n, args.atr_mult)
        else:
            spike_mask = detect_spikes_gk_pct(df, hist_n, args.pct)

        table = monthly_spike_counts(df, spike_mask)
        print(f"=== {name} ===")
        print(table.tail(args.months).to_string())

        recent_spikes = df.index[spike_mask.values][-5:]
        if len(recent_spikes) > 0:
            print(f"最近5次尖峰时间: {[str(t) for t in recent_spikes]}")
        print()


if __name__ == "__main__":
    main()
