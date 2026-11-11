"""
"事件导致短期大幅震荡+连续反向大幅亏损"现象的历史频率/强度分位数，
以及该现象能否被ER（效率比）/路径长度识别。

背景：`vshape_whipsaw_regime.py`验证"换向簇事前市场状态"证伪了用户假设
（事前波动率低更容易触发连续反向亏损），且四档门槛下方向一致、与假设相反
（详见strategy_research_log.md 2026-11-11条目）。用户指出那一轮验证跑偏了
方向——实际要回答的是两个更直接的问题：

1. 当前这种现象（换向簇）本身的频率和强度，在历史上处于什么分位水平？
   （不是"事前市场状态处于什么分位"，是"现象本身的发生密度/损害程度处于
   什么分位"）
2. 这种现象能否被ER低+路径长度大这类指标，在事中识别出来？
   （不是"事前能否预判"，是"事中这段时间的价格行为，和正常交易期的价格
   行为相比有没有系统性差异"）

方法：
1. 复用`vshape_whipsaw_regime.py`里`read_trades_full`/`find_whipsaw_clusters`
   找出全历史换向簇（固定门槛：max_gap_hours=36, min_loss_pct=2.0,
   max_hold_hours=24，对应此前验证里"确有损害但不过严"的口径）。
2. 频率/强度分位数：构造"滚动90天窗口内换向簇次数"和"滚动90天窗口内合计
   亏损"两条随时间滑动的序列（步长1天），再用这条序列本身的3年历史分布，
   给最新值算分位数——这是"现象发生密度"的分位数，不是"事前市场状态"的
   分位数，两者是完全不同的统计对象。
3. ER/路径长度事中对比：换向簇事中窗口（簇起点到簇终点，含首尾两笔交易的
   完整持仓期）内的短窗口ER和路径长度（价格行进总距离/净位移），与同等
   长度的非换向簇窗口（避开所有已知簇窗口的随机抽样）做Mann-Whitney检验，
   回答"这段时间的价格行为本身是否呈现异常的高路径长度/低ER"。

用法：
  python vshape_whipsaw_history_percentile.py
  python vshape_whipsaw_history_percentile.py --dir ~/Downloads/ --min-loss-pct 2.0
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from analysis_core import scan_files_auto
from trend_persistence_signal import (
    efficiency_ratio, SYMBOL_MAP, ASSETS,
    rolling_window_distribution, percentile_of_recent, HIST_WINDOW_DAYS,
)
from vshape_whipsaw_regime import read_trades_full, find_whipsaw_clusters
from market_trend_volatility_check import fetch_ohlcv, BARS_PER_DAY

# 换向簇交易记录最早到2020年，价格数据需要覆盖"3年基线+序列本身的时间跨度"，
# 比 market_trend_volatility_check.py 默认的1500天(约4年)更长
MAX_FETCH_DAYS_LONG = {"1w": 2600, "1d": 2600, "8h": 2600, "4h": 2600, "1h": 2600}

FREQ_STEP_DAYS = 1      # 滑动步长（天）


def build_freq_intensity_series(clusters_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """构造"滚动window_days天窗口内换向簇次数"和"滚动window_days天窗口内
    合计亏损"两条逐日序列。序列本身的时间范围：从最早一个簇的
    cluster_start+window_days天，到最晚一个簇的cluster_start，保证每个
    采样点都有完整的回看窗口。
    """
    starts = clusters_df["cluster_start"].sort_values().reset_index(drop=True)
    pnls = clusters_df.set_index("cluster_start")["combined_pnl_pct"].sort_index()

    t0 = starts.min() + pd.Timedelta(days=window_days)
    t1 = starts.max()
    if t0 >= t1:
        return pd.DataFrame()

    sample_times = pd.date_range(t0, t1, freq=f"{FREQ_STEP_DAYS}D")
    rows = []
    for t in sample_times:
        window_start = t - pd.Timedelta(days=window_days)
        mask = (pnls.index > window_start) & (pnls.index <= t)
        sub = pnls.loc[mask]
        rows.append({"t": t, "count_Nd": len(sub), "sum_pnl_pct_Nd": sub.sum()})
    return pd.DataFrame(rows)


def build_price_path_metrics(symbol: str, timeframe: str, window_days: float) -> dict:
    """返回该标的在指定时间框架下的价格序列、逐bar涨跌幅，供事中窗口ER/
    路径长度计算使用。"""
    df = fetch_ohlcv(symbol, timeframe, MAX_FETCH_DAYS_LONG[timeframe])
    close = df["close"]
    bars_per_day = BARS_PER_DAY[timeframe]
    window_n = max(2, int(round(window_days * bars_per_day)))
    return {"close": close, "window_n": window_n, "bars_per_day": bars_per_day}


def window_er_and_path(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float] | None:
    """给定时间窗口[start, end]，返回该段的ER（净位移/路径长度）和路径长度
    （以相对净位移归一化后的绝对路径百分比，便于跨不同价位的标的比较）。"""
    sub = close.loc[start:end]
    if len(sub) < 3:
        return None
    net_change = abs(sub.iloc[-1] - sub.iloc[0])
    path_sum = sub.diff().abs().sum()
    if path_sum == 0:
        return None
    er = net_change / path_sum
    path_pct = path_sum / sub.iloc[0] * 100  # 路径长度相对起点价格的百分比
    return er, path_pct


def sample_non_cluster_windows(close: pd.Series, cluster_windows: list[tuple], window_n_days: float,
                                n_samples: int, rng: np.random.Generator) -> list[tuple]:
    """在价格序列里随机抽取与换向簇窗口等长的非换向簇时间段（起点避开所有
    已知簇窗口±该窗口长度的缓冲区，避免抽样污染）。"""
    idx = close.index
    total = len(idx)
    window_td = pd.Timedelta(days=window_n_days)
    buffer_td = window_td  # 缓冲区：窗口长度本身，避免抽到紧贴簇边缘的重叠段
    forbidden = [(s - buffer_td, e + buffer_td) for s, e in cluster_windows]

    out = []
    attempts = 0
    max_attempts = n_samples * 50
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        start_pos = rng.integers(0, total - 1)
        start = idx[start_pos]
        end = start + window_td
        if end > idx[-1]:
            continue
        if any(s <= start <= e or s <= end <= e for s, e in forbidden):
            continue
        out.append((start, end))
    return out


def main():
    ap = argparse.ArgumentParser(description="换向簇现象本身的历史频率/强度分位数，及ER/路径长度事中识别力")
    ap.add_argument("--dir", default=os.path.expanduser("~/Downloads/"))
    ap.add_argument("--max-gap-hours", type=float, default=36.0)
    ap.add_argument("--min-loss-pct", type=float, default=2.0,
                     help="两笔都要超过的最小单笔亏损幅度（账户权益百分比绝对值），默认2.0")
    ap.add_argument("--max-hold-hours", type=float, default=24.0)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--n-samples", type=int, default=300, help="非换向簇对照窗口抽样数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freq-window-days", type=int, default=90, help="频率/强度统计的滑动窗口长度（天）")
    args = ap.parse_args()

    files = scan_files_auto(args.dir)
    if not files:
        print(f"在 {args.dir} 未找到策略 xlsx 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：{sorted(files.keys())}")
    print(f"换向簇定义：间隔<={args.max_gap_hours}h，两笔都亏损>={args.min_loss_pct}%，单笔持仓<={args.max_hold_hours}h\n")

    all_clusters = []
    for key, fp in sorted(files.items()):
        asset = next((a for a in ASSETS if key.startswith(a)), None)
        if asset is None:
            continue
        trades = read_trades_full(fp)
        if trades.empty:
            continue
        clusters = find_whipsaw_clusters(trades, args.max_gap_hours, args.min_loss_pct, args.max_hold_hours)
        print(f"{key}: {len(trades)}笔交易，{len(clusters)}个换向簇")
        for c in clusters:
            all_clusters.append({"key": key, "asset": asset, **c})

    clusters_df = pd.DataFrame(all_clusters)
    if clusters_df.empty:
        print("未找到任何换向簇")
        return
    print(f"\n总计 {len(clusters_df)} 个换向簇\n")

    # ============ 问题1：频率/强度的历史分位数 ============
    print("=" * 60)
    print("问题1：换向簇发生密度/损害程度，在历史分布中的分位数")
    print("=" * 60)

    W = args.freq_window_days
    freq_series = build_freq_intensity_series(clusters_df, W)
    if freq_series.empty:
        print("样本时间跨度不足，无法构造频率序列")
    else:
        bars_3y_days = HIST_WINDOW_DAYS
        hist_cutoff = freq_series["t"].max() - pd.Timedelta(days=bars_3y_days)
        hist_part = freq_series[freq_series["t"] >= hist_cutoff] if (freq_series["t"].min() < hist_cutoff) else freq_series

        count_dist = hist_part["count_Nd"].values
        pnl_dist = hist_part["sum_pnl_pct_Nd"].values

        latest = freq_series.iloc[-1]
        count_pct = percentile_of_recent(latest["count_Nd"], count_dist)
        pnl_pct = percentile_of_recent(latest["sum_pnl_pct_Nd"], pnl_dist)

        print(f"频率序列采样点数: {len(freq_series)}（{freq_series['t'].min().date()} ~ {freq_series['t'].max().date()}，步长{FREQ_STEP_DAYS}天，窗口{W}天）")
        print(f"3年基线窗口: {hist_part['t'].min().date()} ~ {hist_part['t'].max().date()}（{len(hist_part)}个采样点）")
        print(f"\n最新值（截至{latest['t'].date()}，往前看{W}天）：")
        print(f"  换向簇次数: {latest['count_Nd']:.0f} 次  历史3年分位数: {count_pct:.1f}%" if count_pct is not None else f"  换向簇次数: {latest['count_Nd']:.0f} 次  历史分位数: 样本不足")
        print(f"  合计亏损: {latest['sum_pnl_pct_Nd']:.2f}%  历史3年分位数: {pnl_pct:.1f}%" if pnl_pct is not None else f"  合计亏损: {latest['sum_pnl_pct_Nd']:.2f}%  历史分位数: 样本不足")

        print(f"\n3年基线分布统计：")
        print(f"  次数({W}d窗口): 均值={count_dist.mean():.2f}  中位数={np.median(count_dist):.1f}  P10={np.percentile(count_dist,10):.1f}  P90={np.percentile(count_dist,90):.1f}  最大={count_dist.max():.0f}")
        print(f"  合计亏损%({W}d窗口): 均值={pnl_dist.mean():.2f}%  中位数={np.median(pnl_dist):.2f}%  P10={np.percentile(pnl_dist,10):.2f}%  P90={np.percentile(pnl_dist,90):.2f}%  最严重={pnl_dist.min():.2f}%")

        print(f"\n近期逐日采样的分位数轨迹（每30天取一个点，看是否近期走高）：")
        for i in range(0, len(freq_series), 30):
            row = freq_series.iloc[i]
            c_pct = percentile_of_recent(row["count_Nd"], count_dist)
            p_pct = percentile_of_recent(row["sum_pnl_pct_Nd"], pnl_dist)
            c_str = f"{c_pct:.0f}%" if c_pct is not None else "N/A"
            p_str = f"{p_pct:.0f}%" if p_pct is not None else "N/A"
            print(f"  {row['t'].date()}: 次数={row['count_Nd']:.0f}(分位{c_str})  合计亏损={row['sum_pnl_pct_Nd']:+.2f}%(分位{p_str})")

    # ============ 问题2：ER/路径长度事中识别力 ============
    print("\n" + "=" * 60)
    print("问题2：换向簇事中窗口 vs 非换向簇窗口，ER/路径长度是否有系统性差异")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)
    cluster_er_rows = []
    baseline_er_rows = []

    for asset in ASSETS:
        symbol = SYMBOL_MAP[asset]
        sub_clusters = clusters_df[clusters_df["asset"] == asset]
        if sub_clusters.empty:
            continue
        print(f"\n拉取 {symbol} 价格数据...")
        close = fetch_ohlcv(symbol, args.timeframe, MAX_FETCH_DAYS_LONG[args.timeframe])["close"]

        cluster_windows = []
        for _, row in sub_clusters.iterrows():
            start, end = row["cluster_start"], row["end"]
            result = window_er_and_path(close, start, end)
            if result is None:
                continue
            er, path_pct = result
            cluster_er_rows.append({"asset": asset, "er": er, "path_pct": path_pct,
                                     "duration_h": (end - start).total_seconds() / 3600})
            cluster_windows.append((start, end))

        if not cluster_windows:
            continue
        avg_window_days = np.mean([(e - s).total_seconds() / 86400 for s, e in cluster_windows])
        non_cluster_windows = sample_non_cluster_windows(close, cluster_windows, avg_window_days, args.n_samples, rng)
        for start, end in non_cluster_windows:
            result = window_er_and_path(close, start, end)
            if result is None:
                continue
            er, path_pct = result
            baseline_er_rows.append({"asset": asset, "er": er, "path_pct": path_pct})

    cluster_er_df = pd.DataFrame(cluster_er_rows)
    baseline_er_df = pd.DataFrame(baseline_er_rows)

    if cluster_er_df.empty or baseline_er_df.empty:
        print("样本不足，无法做ER/路径长度对比")
        return

    print(f"\n换向簇事中窗口样本数: {len(cluster_er_df)}  平均持续时长: {cluster_er_df['duration_h'].mean():.1f}小时")
    print(f"非换向簇对照窗口样本数: {len(baseline_er_df)}")

    print(f"\n换向簇事中窗口:")
    print(f"  ER: 均值={cluster_er_df['er'].mean():.3f}  中位数={cluster_er_df['er'].median():.3f}")
    print(f"  路径长度%: 均值={cluster_er_df['path_pct'].mean():.2f}%  中位数={cluster_er_df['path_pct'].median():.2f}%")

    print(f"\n非换向簇对照窗口（同等时长随机抽样）:")
    print(f"  ER: 均值={baseline_er_df['er'].mean():.3f}  中位数={baseline_er_df['er'].median():.3f}")
    print(f"  路径长度%: 均值={baseline_er_df['path_pct'].mean():.2f}%  中位数={baseline_er_df['path_pct'].median():.2f}%")

    print(f"\nMann-Whitney U检验：")
    for col, label in [("er", "ER"), ("path_pct", "路径长度%")]:
        a = cluster_er_df[col].dropna().values
        b = baseline_er_df[col].dropna().values
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        direction = "换向簇更低" if np.median(a) < np.median(b) else "换向簇更高"
        print(f"  {label}: p={p:.6f}  {'显著' if p < 0.05 else '不显著'}  {direction}")

    # ROC式区分力：用路径长度%作为判别换向簇 vs 非换向簇的能力
    print(f"\n用路径长度%单指标区分换向簇 vs 非换向簇窗口的AUC：")
    labels = np.concatenate([np.ones(len(cluster_er_df)), np.zeros(len(baseline_er_df))])
    scores = np.concatenate([cluster_er_df["path_pct"].values, baseline_er_df["path_pct"].values])
    valid = ~np.isnan(scores)
    auc = stats.mannwhitneyu(scores[valid][labels[valid] == 1], scores[valid][labels[valid] == 0],
                              alternative="two-sided").statistic / (labels[valid].sum() * (1 - labels[valid]).sum())
    print(f"  路径长度% AUC = {auc:.3f}（0.5=无区分力，1.0=完美区分，<0.5等价于反向指标）")

    print(f"\n用ER单指标区分换向簇 vs 非换向簇窗口的AUC：")
    scores_er = np.concatenate([cluster_er_df["er"].values, baseline_er_df["er"].values])
    valid_er = ~np.isnan(scores_er)
    auc_er = stats.mannwhitneyu(scores_er[valid_er][labels[valid_er] == 1], scores_er[valid_er][labels[valid_er] == 0],
                                 alternative="two-sided").statistic / (labels[valid_er].sum() * (1 - labels[valid_er]).sum())
    print(f"  ER AUC = {auc_er:.3f}（如果换向簇ER更低，AUC应<0.5；为方便解读显示 1-AUC={1-auc_er:.3f}）")


if __name__ == "__main__":
    main()
