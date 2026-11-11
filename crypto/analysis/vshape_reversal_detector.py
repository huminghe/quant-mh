"""
V型折返结构检测：短窗口效率比（ER），弥补单bar尖峰检测的盲区。

背景：2026-11-11 `spike_bar_detector.py`验证的"门结构"指标（ATR倍数法/GK历史
分位数法）只能检测单根K线内部的异常波动。用户指出存在另一种现象——连续多根
K线方向相反、净位移小但路径长度大的"V型/倒V型"往返（如2026-11-08 20:00~
11-09 12:00 BTC：76168→73333→75500，没有任何单根bar跌幅超过1.5%，但36小时
内先跌3.7%再涨2.9%），这种结构不会被单bar尖峰检测捕捉到。

核心指标：Kaufman效率比ER = |净位移| / 路径长度（已存在于
`trend_persistence_signal.py`，此前只用30天月度窗口衡量趋势性）。本脚本用
短窗口（n=12根4h bar=48小时）复用同一函数，验证能否识别V型折返。

验证结论（2026-11-11，详见strategy_research_log.md同日两条条目）：
1. 短窗口ER能捕捉已知5次门结构事件的V型特征——20组"标的×事件"组合中16组
   历史分位数<20%（多数1-10%），4个标的在n=12窗口下均在事件窗口内触现局部低点。
2. 但特异性不足：按10%分位阈值，近90天每个标的约产生40次独立低值段，而已知
   造成策略连续亏损的门结构事件只有5次——指标对"净位移小路径长"高度敏感，
   无法区分"轻微往返"和"剧烈到冲击策略"的V型。
3. 补做绝对值门槛验证（用户指出ER是纯比值，分不清小幅磨损和大幅冲击）：
   改用止损距离（3.5倍ATR20，对齐已集成的8H Chandelier Exit参数）做绝对值
   门槛，单向穿透版本漏判11-08/11-10两次事件（止损距离本身没被打穿），双向
   折返版本反而漏判11-05这次最剧烈的单边暴跌，两种变体的近90天误报率
   （22-40次 vs 已知5次）都没有改善。根因不是口径或门槛数值选得不准——路径
   长度/振幅能准确衡量"折返的物理尺度"，但"物理尺度大"和"是否真正伤害策略"
   是两个不完全重合的问题，这中间的映射关系本身不稳定。

**结论性质与门结构单bar检测一致：只能用于事后归因，不能作为提前规避或实时
预警的交易规则**——指标无前瞻性验证，且高误报率（约40:5，绝对值门槛版本
22-40:5）意味着任何基于此阈值的实时触发机制都会被大量非损害性的正常波动
噪声淹没。ER+绝对值门槛两个方向均已验证到头，详见strategy_research_log.md
2026-11-11"V型折返绝对值门槛三轮验证"条目（含止损方案事前规避能力的调研
结论）。

用法：
  python vshape_reversal_detector.py --timeframe 4h --window 12
  python vshape_reversal_detector.py --timeframe 4h --window 12 --events 2026-10-29,2026-11-04
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from scipy import stats

from trend_persistence_signal import efficiency_ratio, SYMBOL_MAP, ASSETS, HIST_WINDOW_DAYS
from market_trend_volatility_check import fetch_ohlcv, BARS_PER_DAY, MAX_FETCH_DAYS

DEFAULT_EVENTS = ["2026-10-29", "2026-11-04", "2026-11-05", "2026-11-08", "2026-11-10"]


def event_window_er(er: pd.Series, event_date: str, pre_hours: int = 12, post_hours: int = 36) -> tuple[float, pd.Timestamp] | None:
    """事件日前后窗口内ER最小值及其时间点。"""
    start = pd.Timestamp(event_date) - pd.Timedelta(hours=pre_hours)
    end = pd.Timestamp(event_date) + pd.Timedelta(hours=post_hours)
    sub = er.loc[start:end].dropna()
    if sub.empty:
        return None
    return sub.min(), sub.idxmin()


def independent_low_segments(low_mask: pd.Series) -> int:
    """低值bar合并成连续段后计数（避免同一次事件内多根bar重复计数为多次）。"""
    groups = (low_mask != low_mask.shift()).cumsum()
    segs = low_mask.groupby(groups).agg(["sum"])
    return int((segs["sum"] > 0).sum())


def main():
    ap = argparse.ArgumentParser(description="短窗口效率比检测V型折返结构")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--window", type=int, default=12, help="ER窗口bar数，默认12根4h=48小时")
    ap.add_argument("--pct", type=float, default=10.0, help="判定'折返剧烈'的历史分位阈值")
    ap.add_argument("--events", default=",".join(DEFAULT_EVENTS), help="逗号分隔的已知事件日期，用于交叉验证")
    args = ap.parse_args()

    tf = args.timeframe
    n = args.window
    events = args.events.split(",")
    bars_per_day = BARS_PER_DAY[tf]
    bars_3y = int(HIST_WINDOW_DAYS * bars_per_day)

    print(f"时间框架: {tf}  ER窗口: {n}根bar(约{n * 24 // bars_per_day}小时)  判定阈值: 3年滚动分布第{args.pct}分位\n")

    for name in ASSETS:
        symbol = SYMBOL_MAP[name]
        df = fetch_ohlcv(symbol, tf, MAX_FETCH_DAYS[tf])
        er = efficiency_ratio(df["close"], n).dropna()
        hist3y = er.tail(bars_3y)
        threshold = hist3y.quantile(args.pct / 100)

        print(f"=== {name}（ER 3年{args.pct}分位阈值={threshold:.3f}） ===")
        for ev in events:
            result = event_window_er(er, ev)
            if result is None:
                print(f"  {ev}: 无数据")
                continue
            minval, mintime = result
            rest = hist3y.drop(hist3y.index.intersection(er.loc[mintime - pd.Timedelta(hours=48):mintime + pd.Timedelta(hours=48)].index), errors="ignore")
            pctile = stats.percentileofscore(rest.values, minval, kind="mean") if len(rest) >= 10 else None
            flag = " <-- 命中(低于阈值)" if minval < threshold else ""
            pctile_str = f"{pctile:.1f}%" if pctile is not None else "样本不足"
            print(f"  {ev}: ER_min={minval:.3f} @ {mintime}  历史分位={pctile_str}{flag}")

        low_mask = er < threshold
        n_all = independent_low_segments(low_mask)
        n_recent90d = independent_low_segments(low_mask.tail(int(90 * bars_per_day)))
        days_total = len(er) / bars_per_day
        print(f"  全历史独立低值段: {n_all}（{days_total:.0f}天，折算每90天约{n_all / days_total * 90:.1f}次）")
        print(f"  近90天独立低值段: {n_recent90d}  （已知事件数: {len(events)}，误报率约{n_recent90d}:{len(events)}）")
        print()

    print("⚠️  结论：该指标能捕捉V型折返特征，但特异性不足（近90天误报约40次 vs 已知事件5次），")
    print("   不能直接作为实时预警/交易过滤规则，仅可用于事后归因。叠加路径长度绝对阈值无法解决")
    print("   特异性问题（会漏判路径长度较小的真实事件，如2026-11-08），不建议采用双重过滤方案。")


if __name__ == "__main__":
    main()
