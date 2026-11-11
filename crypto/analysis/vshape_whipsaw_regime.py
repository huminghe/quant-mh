"""
V型折返是否伤害策略：取决于"折返发生之前"市场的波动率/趋势性状态？

背景：2026-11-11 `vshape_reversal_detector.py`证明短窗口ER+绝对值门槛都无法
从价格形态本身区分"无害折返"和"伤害策略的折返"（详见strategy_research_log.md
同日"V型折返绝对值门槛三轮验证"条目）。用户提出新假设：折返是否演变成"频繁
换向+连续反向大幅亏损"，取决于折返发生**之前**市场处于什么状态——

1. 事前已经波动性启动、趋势性强：折返只是趋势内的正常扰动，不易打断持仓方向。
2. 事前波动率低、没有明确方向（盘整蓄势）：折返更容易导致频繁换向，且换向后
   很快再反转，造成连续多笔亏损。

本脚本v2（11-11第二版）相对第一版的两个关键修正，均由用户指出：

1. **事前窗口不能与事件本身重叠**：第一版`regime_at`用换向簇mid时刻为右端点
   往回看30天，这个窗口本身覆盖了换向簇发生的时间段，统计的是"折返期间"而
   非"折返之前"的市场状态，这是因果关系判断的致命缺陷。v2改为：查询窗口严格
   止于换向簇**起点**（第一笔交易入场时间）之前，窗口长度7天，不与簇本身有
   任何时间重叠。
2. **换向簇定义未区分"事件型快速反转"和"正常趋势止损"**：第一版只看方向、
   gap、盈亏门槛，混入了大量持仓数天的正常趋势止损案例（多见于2021年极端
   行情），稀释了"事件导致短时间内连续反向亏损"这个具体现象的统计信号。v2
   加入单笔持仓周期上限24小时的过滤。

用法：
  python vshape_whipsaw_regime.py
  python vshape_whipsaw_regime.py --dir ~/Downloads/ --max-gap-hours 36 --max-hold-hours 24
"""
import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import openpyxl
import pandas as pd

from analysis_core import scan_files_auto
from trend_persistence_signal import (
    efficiency_ratio, adx, SYMBOL_MAP, ASSETS,
    garman_klass_variance_series, rolling_window_distribution, percentile_of_recent,
)
from market_trend_volatility_check import fetch_ohlcv, BARS_PER_DAY, MAX_FETCH_DAYS

TZ_OFFSET_HOURS = 8    # TV导出为UTC+8 naive，需减8小时对齐Binance UTC数据
PRE_WINDOW_DAYS = 7    # 事前窗口长度：查询"折返发生之前"市场状态回看多少天
HIST_WINDOW_DAYS = 365 * 3  # 历史基线：滚动3年


def read_trades_full(filepath: str) -> pd.DataFrame:
    """读取逐笔交易，返回每笔完整交易的方向/入场出场时间(UTC)/盈亏。"""
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = None
    for name in ["交易", "交易清单", "List of trades", "Trades"]:
        if name in wb.sheetnames:
            ws = wb[name]
            break
    if ws is None:
        wb.close()
        return pd.DataFrame()

    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    recs = {}
    for r in rows[1:]:
        trade_id, ttype, dt, signal, price, *_rest = r
        profit = r[7]
        profit_pct = r[8]
        if trade_id is None or not isinstance(dt, datetime.datetime):
            continue
        rec = recs.setdefault(trade_id, {})
        if "进场" in str(ttype):
            rec["entry_dt"] = dt
            rec["direction"] = "long" if "多头" in str(ttype) else "short"
        elif "出场" in str(ttype):
            rec["exit_dt"] = dt
            rec["profit"] = float(profit) if profit is not None else None
            rec["profit_pct"] = float(profit_pct) if profit_pct is not None else None

    out = []
    for tid, rec in recs.items():
        if "entry_dt" in rec and "exit_dt" in rec and rec.get("profit") is not None:
            out.append({
                "trade_id": tid,
                "direction": rec["direction"],
                "entry_dt": rec["entry_dt"] - pd.Timedelta(hours=TZ_OFFSET_HOURS),
                "exit_dt": rec["exit_dt"] - pd.Timedelta(hours=TZ_OFFSET_HOURS),
                "profit": rec["profit"],
                "profit_pct": rec["profit_pct"],
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("entry_dt").reset_index(drop=True)


def find_whipsaw_clusters(trades: pd.DataFrame, max_gap_hours: float, min_loss_pct: float,
                           max_hold_hours: float) -> list[dict]:
    """找"事件型"换向簇：方向与前一笔不同、入场时间与前一笔出场时间间隔
    <=max_gap_hours、两笔亏损幅度都超过min_loss_pct门槛、且两笔单笔持仓周期都
    <=max_hold_hours。

    持仓周期上限用于排除正常趋势跟踪里持仓数天后止损被打的案例（历史上大量
    出现在2021年极端行情，性质是"趋势判断本身没错，只是最终反转止损"，不是
    用户描述的"短时间内事件导致连续反向大幅亏损"）。
    """
    clusters = []
    for i in range(1, len(trades)):
        prev, cur = trades.iloc[i - 1], trades.iloc[i]
        if cur["direction"] == prev["direction"]:
            continue
        gap_hours = (cur["entry_dt"] - prev["exit_dt"]).total_seconds() / 3600
        if gap_hours < 0 or gap_hours > max_gap_hours:
            continue
        if prev["profit_pct"] >= -min_loss_pct or cur["profit_pct"] >= -min_loss_pct:
            continue
        prev_hold_h = (prev["exit_dt"] - prev["entry_dt"]).total_seconds() / 3600
        cur_hold_h = (cur["exit_dt"] - cur["entry_dt"]).total_seconds() / 3600
        if prev_hold_h > max_hold_hours or cur_hold_h > max_hold_hours:
            continue
        clusters.append({
            "cluster_start": prev["entry_dt"],  # 簇起点：第一笔交易的入场时间，事前窗口右端点对齐于此
            "mid": prev["exit_dt"],              # 换向发生的大致时刻
            "end": cur["exit_dt"],
            "combined_pnl": prev["profit"] + cur["profit"],
            "combined_pnl_pct": prev["profit_pct"] + cur["profit_pct"],
            "prev_hold_h": prev_hold_h, "cur_hold_h": cur_hold_h,
        })
    return clusters


def build_regime_series(symbol: str, timeframe: str) -> dict:
    """构建该标的在指定框架下的ADX/ER/GK波动率的7天滚动窗口均值序列，及对应的
    3年滚动分位数分布基准（口径：分布样本与查询值必须都是7天窗口均值，不能
    混用瞬时值，否则分位数系统性失真）。
    """
    df = fetch_ohlcv(symbol, timeframe, MAX_FETCH_DAYS[timeframe])
    bars_per_day = BARS_PER_DAY[timeframe]
    bars_3y = int(HIST_WINDOW_DAYS * bars_per_day)
    window_n = max(2, int(PRE_WINDOW_DAYS * bars_per_day))

    adx14 = adx(df, 14)
    er_n = efficiency_ratio(df["close"], window_n)
    gk_var = garman_klass_variance_series(df)
    periods_per_year = 365 * bars_per_day

    adx_roll = adx14.rolling(window_n).mean()
    er_roll = er_n.rolling(window_n).mean()
    vol_roll = np.sqrt(gk_var.rolling(window_n).mean() * periods_per_year) * 100

    adx_hist_dist = rolling_window_distribution(adx14.dropna().tail(bars_3y), window_n, np.mean)
    er_hist_dist = rolling_window_distribution(er_n.dropna().tail(bars_3y), window_n, np.mean)
    vol_hist_dist = rolling_window_distribution(
        np.sqrt(gk_var.dropna().tail(bars_3y) * periods_per_year) * 100, window_n, np.mean)

    return {
        "adx": adx_roll, "er": er_roll, "vol": vol_roll,
        "adx_dist": adx_hist_dist, "er_dist": er_hist_dist, "vol_dist": vol_hist_dist,
    }


def regime_before(regime: dict, cluster_start: pd.Timestamp, bar_duration: pd.Timedelta) -> dict | None:
    """取"cluster_start之前"最近一个7天窗口均值（ffill到严格早于cluster_start
    的最后一个bar），换算成3年滚动分布分位数。不允许使用cluster_start及之后
    的任何数据，保证窗口与换向簇本身的发生时间不重叠。

    `series`的索引是K线**开始**时间，一根bar在其开始时间到结束时间之间都
    未收线。若直接用`series.index < cluster_start`，当cluster_start落在
    某根bar内部（而非恰好对齐bar网格整点，这是常态）时，会选中这根尚未
    收线、实际跨度覆盖到cluster_start之后的bar，造成最长一个bar周期的
    未来数据泄漏（2026-11-11复核发现）。改为比较bar的结束时间
    （index + bar_duration）与cluster_start，确保只使用严格在
    cluster_start之前已经收线完成的bar。
    """
    out = {}
    for key, dist_key in [("adx", "adx_dist"), ("er", "er_dist"), ("vol", "vol_dist")]:
        series = regime[key].dropna()
        before = series.loc[series.index + bar_duration <= cluster_start]
        if before.empty:
            return None
        val = float(before.iloc[-1])
        pctile = percentile_of_recent(val, regime[dist_key])
        out[f"{key}_val"] = val
        out[f"{key}_pct"] = pctile
    return out


def main():
    ap = argparse.ArgumentParser(description="V型折返演变成连续反向亏损，是否取决于事前市场状态")
    ap.add_argument("--dir", default=os.path.expanduser("~/Downloads/"))
    ap.add_argument("--max-gap-hours", type=float, default=36.0,
                     help="两笔反向交易之间最大间隔（出场到下一笔入场），默认36小时")
    ap.add_argument("--min-loss-pct", type=float, default=1.0,
                     help="两笔都要超过的最小单笔亏损幅度（账户权益百分比绝对值），默认1.0")
    ap.add_argument("--max-hold-hours", type=float, default=24.0,
                     help="单笔持仓周期上限（小时），默认24，排除正常趋势跟踪的多日止损案例")
    ap.add_argument("--timeframe", default="4h", help="市场状态指标计算用的K线框架")
    args = ap.parse_args()

    files = scan_files_auto(args.dir)
    if not files:
        print(f"在 {args.dir} 未找到策略 xlsx 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：{sorted(files.keys())}")
    print(f"换向簇定义：连续反向交易，间隔<={args.max_gap_hours}小时，两笔都亏损>={args.min_loss_pct}%，"
          f"且单笔持仓<={args.max_hold_hours}小时（排除多日正常趋势止损）")
    print(f"事前窗口：换向簇起点（第一笔入场时间）往前{PRE_WINDOW_DAYS}天，不与簇本身重叠\n")

    regime_cache: dict[str, dict] = {}
    all_clusters = []
    all_trade_entries = []  # 全部交易入场时间，作为对照基线（同样用事前窗口，不用mid）

    for key, fp in sorted(files.items()):
        asset = next((a for a in ASSETS if key.startswith(a)), None)
        if asset is None:
            continue
        trades = read_trades_full(fp)
        if trades.empty:
            continue

        clusters = find_whipsaw_clusters(trades, args.max_gap_hours, args.min_loss_pct, args.max_hold_hours)
        print(f"{key}: {len(trades)}笔交易，{len(clusters)}个事件型换向簇")
        for c in clusters:
            all_clusters.append((key, asset, c))
        for ts in trades["entry_dt"]:
            all_trade_entries.append((asset, ts))

    print(f"\n总计 {len(all_clusters)} 个事件型换向簇（跨{len([f for f in files if any(f.startswith(a) for a in ASSETS)])}个策略，可能有重叠时间段）\n")

    print("构建市场状态基准序列（ADX/ER/GK波动率，7天窗口，滚动3年分位数）...")
    for asset in ASSETS:
        regime_cache[asset] = build_regime_series(SYMBOL_MAP[asset], args.timeframe)

    bar_duration = pd.Timedelta(days=1) / BARS_PER_DAY[args.timeframe]

    cluster_rows = []
    for key, asset, c in all_clusters:
        r = regime_before(regime_cache[asset], c["cluster_start"], bar_duration)
        if r is None:
            continue
        r.update({"key": key, "asset": asset, "cluster_start": c["cluster_start"],
                  "combined_pnl": c["combined_pnl"], "combined_pnl_pct": c["combined_pnl_pct"]})
        cluster_rows.append(r)

    baseline_rows = []
    for asset, ts in all_trade_entries:
        r = regime_before(regime_cache[asset], ts, bar_duration)
        if r is None:
            continue
        r["asset"] = asset
        baseline_rows.append(r)

    cluster_df = pd.DataFrame(cluster_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    if cluster_df.empty:
        print("未找到任何事件型换向簇，无法验证假设")
        return

    print(f"\n=== 事件型换向簇列表（n={len(cluster_df)}，按时间排序） ===")
    for _, row in cluster_df.sort_values("cluster_start").iterrows():
        print(f"  {row['cluster_start']}  {row['key']:16s}  合计亏损{row['combined_pnl_pct']:+.2f}%（{row['combined_pnl']:+.1f} USDT）  "
              f"事前7天: ADX分位={row['adx_pct']}  ER分位={row['er_pct']}  波动率分位={row['vol_pct']}")

    print(f"\n=== 事件型换向簇发生前7天的市场状态分位数（n={len(cluster_df)}） ===")
    for col in ["adx_pct", "er_pct", "vol_pct"]:
        vals = cluster_df[col].dropna()
        print(f"  {col}: 均值={vals.mean():.1f}  中位数={vals.median():.1f}  n={len(vals)}  "
              f"<25分位占比={100*(vals < 25).mean():.1f}%  >75分位占比={100*(vals > 75).mean():.1f}%")

    print(f"\n=== 全部交易入场前7天（对照基线）的市场状态分位数（n={len(baseline_df)}） ===")
    for col in ["adx_pct", "er_pct", "vol_pct"]:
        vals = baseline_df[col].dropna()
        print(f"  {col}: 均值={vals.mean():.1f}  中位数={vals.median():.1f}  n={len(vals)}  "
              f"<25分位占比={100*(vals < 25).mean():.1f}%  >75分位占比={100*(vals > 75).mean():.1f}%")

    from scipy import stats
    print("\n=== 事件型换向簇 vs 全部交易：Mann-Whitney U检验（事前市场状态分位数是否有系统性差异） ===")
    for col in ["adx_pct", "er_pct", "vol_pct"]:
        a = cluster_df[col].dropna().values
        b = baseline_df[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            print(f"  {col}: 样本不足")
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        direction = "换向簇事前分位数更低(偏盘整)" if np.median(a) < np.median(b) else "换向簇事前分位数更高(偏趋势/高波动)"
        print(f"  {col}: p={p:.4f}  {'显著' if p < 0.05 else '不显著'}  {direction}")

    print(f"\n事件型换向簇总亏损: {cluster_df['combined_pnl'].sum():.1f} USDT（{len(cluster_df)}个簇）")

    # 频率与强度：按日历时间折算发生频率，给出强度分布
    span_days = (cluster_df["cluster_start"].max() - cluster_df["cluster_start"].min()).days
    print(f"\n=== 频率与强度 ===")
    print(f"数据覆盖时间跨度: {span_days}天（{cluster_df['cluster_start'].min().date()} ~ {cluster_df['cluster_start'].max().date()}）")
    print(f"事件型换向簇总数: {len(cluster_df)}，平均每90天发生 {len(cluster_df) / span_days * 90:.2f} 次")
    print(f"合计亏损%分布: 中位数={cluster_df['combined_pnl_pct'].median():.2f}%  "
          f"P25={cluster_df['combined_pnl_pct'].quantile(0.25):.2f}%  P75={cluster_df['combined_pnl_pct'].quantile(0.75):.2f}%  "
          f"最严重={cluster_df['combined_pnl_pct'].min():.2f}%")


if __name__ == "__main__":
    main()
