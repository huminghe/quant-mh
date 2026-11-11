"""
门结构（单bar尖峰）月度频率的历史分位数——判断"近期尖峰变多"是否异常。

背景：2026-11-11发现9个部署策略约51%近期亏损来自少数门结构事件。用户
进一步问：这种短期大幅震荡本身，近期出现的频率相比历史是否异常，还是
历史上一直都会周期性出现？需要一个能回答"近期尖峰频率处于历史什么水位"
的指标，延续market_trend_methodology_2026_11.md第5节"滚动3年+分位数"
的方法论框架，只是把统计对象从ADX/ER/波动率均值换成"月度尖峰bar数"。

判定方式：直接用单bar收益率绝对值（不依赖GK方差的参数设定——2026-11-11
testing发现GK 95分位在8h框架上会漏判肉眼明显的尖峰，需要换一个不依赖
额外参数校准的量，多方法交叉验证更可信）。阈值用全部可用历史（≤4年）
单bar绝对收益率分布的第pct分位，与月度统计用同一份数据保证口径一致。

用法：
  python spike_frequency_monthly.py --timeframe 4h --months 36 --pct 95
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from trend_persistence_signal import ASSETS, SYMBOL_MAP, HIST_WINDOW_DAYS
from market_trend_volatility_check import fetch_ohlcv, BARS_PER_DAY, MAX_FETCH_DAYS


def monthly_spike_freq(df: pd.DataFrame, pct: float, bars_per_day: float) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """返回(每月尖峰bar数, 每30天折算频率, 月度最大单bar幅度%, 全历史阈值%)。
    阈值用全部可用数据算一次，不是滚动基线——这里要回答的是"月度频率/幅度
    在全历史里排第几"，用固定阈值+按月统计，再对"频率/幅度序列"做跨月分位数对比。
    次数按实际天数折算成"每30天频率"，避免当月未走完时次数被低估（11月至今
    只有11/30天，直接比次数对不完整月份不公平）。
    """
    ret_pct = df["close"].pct_change().abs() * 100
    threshold = ret_pct.quantile(pct / 100)
    spike_mask = ret_pct > threshold
    month = df.index.to_series().dt.to_period("M")
    counts = month[spike_mask.values].groupby(month[spike_mask.values]).count()
    full_months = month.unique()
    counts = counts.reindex(full_months, fill_value=0).sort_index()

    bars_total = month.groupby(month).count().reindex(full_months, fill_value=0)
    days_in_month = bars_total / bars_per_day
    freq_per_30d = (counts / days_in_month * 30).round(1)

    max_ret = ret_pct.groupby(month).max().reindex(full_months)
    return counts, freq_per_30d, max_ret, threshold


def main():
    ap = argparse.ArgumentParser(description="单bar尖峰月度频率的历史分位数")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--months", type=int, default=36, help="展示最近几个月")
    ap.add_argument("--pct", type=float, default=95.0, help="尖峰判定用的历史分位阈值")
    args = ap.parse_args()

    tf = args.timeframe
    print(f"时间框架: {tf}  尖峰判定: 单bar|收益%| > 全历史第{args.pct}分位\n")

    def pctile_excl_last(series, value, idx, last_idx):
        hist = series.iloc[:-1] if idx == last_idx else series
        hist = hist.dropna()
        return stats.percentileofscore(hist.values, value, kind="mean") if len(hist) >= 10 else None

    all_freq = {}
    all_maxret = {}
    for name in ASSETS:
        symbol = SYMBOL_MAP[name]
        df = fetch_ohlcv(symbol, tf, MAX_FETCH_DAYS[tf])
        counts, freq_per_30d, max_ret, threshold = monthly_spike_freq(df, args.pct, BARS_PER_DAY[tf])
        all_freq[name] = freq_per_30d
        all_maxret[name] = max_ret
        print(f"=== {name}（尖峰阈值 |单bar收益%| > {threshold:.2f}%） ===")

        last_idx = freq_per_30d.index[-1]
        recent = freq_per_30d.tail(args.months)
        for ym in recent.index:
            cnt, freq = counts.loc[ym], freq_per_30d.loc[ym]
            mret = max_ret.loc[ym]
            freq_pctile = pctile_excl_last(freq_per_30d, freq, ym, last_idx)
            mret_pctile = pctile_excl_last(max_ret, mret, ym, last_idx)
            flag = ""
            if freq_pctile is not None and freq_pctile > 75:
                flag += " <-- 频率历史高位"
            if mret_pctile is not None and mret_pctile > 75:
                flag += " <-- 单日最大波动历史高位"
            freq_note = f"{freq_pctile:.0f}%分位" if freq_pctile is not None else "样本不足"
            mret_note = f"{mret_pctile:.0f}%分位" if mret_pctile is not None else "样本不足"
            print(f"  {ym}: {cnt:>2}次(原始) {freq:>5.1f}次/30天({freq_note})  "
                  f"月内最大单bar{mret:>5.2f}%({mret_note}){flag}")
        print()

    print("===== 跨标的合计（折算每30天频率） =====")
    combined_freq = pd.DataFrame(all_freq).fillna(0)
    combined_freq["total"] = combined_freq.sum(axis=1)
    combined_maxret = pd.DataFrame(all_maxret)
    combined_maxret["max"] = combined_maxret.max(axis=1)
    last_idx = combined_freq.index[-1]
    recent_total = combined_freq["total"].tail(args.months)
    for ym in recent_total.index:
        freq = combined_freq["total"].loc[ym]
        mret = combined_maxret["max"].loc[ym]
        freq_pctile = pctile_excl_last(combined_freq["total"], freq, ym, last_idx)
        mret_pctile = pctile_excl_last(combined_maxret["max"], mret, ym, last_idx)
        flag = ""
        if freq_pctile is not None and freq_pctile > 75:
            flag += " <-- 频率历史高位"
        if mret_pctile is not None and mret_pctile > 75:
            flag += " <-- 单日最大波动历史高位"
        freq_note = f"{freq_pctile:.0f}%分位" if freq_pctile is not None else "样本不足"
        mret_note = f"{mret_pctile:.0f}%分位" if mret_pctile is not None else "样本不足"
        print(f"  {ym}: {freq:>5.1f}次/30天（4标的合计，{freq_note}）  "
              f"4标的月内最大单bar{mret:>5.2f}%（{mret_note}）{flag}")


if __name__ == "__main__":
    main()
