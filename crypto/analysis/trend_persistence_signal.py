"""
趋势持续性指标（Hurst/ADX/效率比）区分力验证

背景：2026-10-21健康度监控调研证伪了"用策略PnL统计量（盈亏比/胜率/Sharpe/
MFE/CUSUM）触发暂停"这条路线，根因是这类指标全是结果性/滞后的。调研后续
方向1是"不测策略PnL，测市场/价格本身的趋势持续性，连续化调整仓位"，与已
上线的ATR定仓（方法A，风险平衡层，按波动率水平调整仓位）是不同的信号维度。

本脚本只做第一步：验证趋势持续性指标本身有没有区分力，不涉及仓位决策。
用价格序列（不是策略PnL）计算三个候选指标的月度值，与各策略同期月度PnL
对齐，看指标取值和策略表现是否有可辨识关系。

三个候选指标（全部基于日线收盘价，窗口30天滚动，月末采样）：
  - Hurst exponent：R/S分析，>0.5趋势持续，<0.5均值回归，=0.5随机游走
  - ADX(14)：>25视为"有趋势"，<20视为"无趋势/盘整"，标准技术指标口径
  - 效率比 Kaufman ER：|净价格变动| / sum(|逐日变动|)，1=完美趋势，0=纯噪音

用法：
  python trend_persistence_signal.py --dir ~/Downloads/
"""
import argparse
import datetime
import statistics
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import ccxt

from analysis_core import scan_files_auto
from rolling_pl_ratio import read_trade_pnls
from trigger_grid_scan import monthly_pnl_series, drop_incomplete_last_month

DEPLOYED_SUFFIXES = ("_v2", "_v3_205m")
ASSETS = ["BTC", "ETH", "SOL", "DOGE"]
SYMBOL_MAP = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "DOGE": "DOGE/USDT"}

ROLL_WINDOW_DAYS = 30  # 指标滚动窗口，月末采样对齐月度PnL


def fetch_daily_ohlcv(symbol: str) -> pd.DataFrame:
    print(f"  拉取 {symbol} 日线...", end=" ", flush=True)
    ex = ccxt.binance({"options": {"defaultType": "future"}, "timeout": 30000})
    all_bars, since = [], ex.parse8601("2019-01-01T00:00:00Z")
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
                break
            except Exception:
                if attempt == 2:
                    raise
                import time
                time.sleep(3)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < 1000:
            break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("dt").sort_index()
    print(f"done ({len(df)} bars)")
    return df


def hurst_exponent(series: np.ndarray) -> float | None:
    """R/S分析估计Hurst指数，series为对数收益率。"""
    n = len(series)
    if n < 20:
        return None
    lags = range(2, min(n // 2, 20))
    tau = []
    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        if len(diffs) < 2:
            continue
        std = np.std(diffs)
        tau.append(std if std > 0 else np.nan)
    if len(tau) < 4:
        return None
    tau = np.array(tau)
    valid = ~np.isnan(tau)
    if valid.sum() < 4:
        return None
    log_lags = np.log(list(lags)[: len(tau)])[valid]
    log_tau = np.log(tau[valid])
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return slope  # H = slope（对数差分标准差与lag的幂律关系）


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    net_change = (close - close.shift(n)).abs()
    path_sum = close.diff().abs().rolling(n).sum()
    return net_change / path_sum


def monthly_signal_series(df: pd.DataFrame) -> list[tuple]:
    """按月末采样三个指标，返回[(年月, {hurst, adx, er}), ...]。"""
    log_ret = np.log(df["close"]).diff().dropna()
    adx_series = adx(df, 14)
    er_series = efficiency_ratio(df["close"], ROLL_WINDOW_DAYS)

    month_ends = df.resample("ME").last().index
    results = []
    for me in month_ends:
        window_end_idx = df.index.get_indexer([me], method="pad")[0]
        if window_end_idx < ROLL_WINDOW_DAYS:
            continue
        window_log_ret = log_ret.iloc[max(0, window_end_idx - ROLL_WINDOW_DAYS): window_end_idx].values
        h = hurst_exponent(window_log_ret)
        a = adx_series.iloc[window_end_idx]
        e = er_series.iloc[window_end_idx]
        ym = (me.year, me.month)
        results.append((ym, {"hurst": h, "adx": a if pd.notna(a) else None, "er": e if pd.notna(e) else None}))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/Downloads/")
    args = parser.parse_args()
    directory = args.dir.replace("~", str(Path.home()))

    files = scan_files_auto(directory)
    deployed = {k: v for k, v in files.items() if k.endswith(DEPLOYED_SUFFIXES)}

    print("===== 趋势持续性指标 vs 策略月度PnL：描述性关系验证 =====\n")

    for asset in ASSETS:
        symbol = SYMBOL_MAP[asset]
        df = fetch_daily_ohlcv(symbol)
        signal_series = monthly_signal_series(df)
        signal_map = {ym: v for ym, v in signal_series}

        strat_keys = [k for k in deployed if k.startswith(asset + "_")]
        for key in strat_keys:
            trades = read_trade_pnls(deployed[key])
            if not trades:
                continue
            monthly_pnl = monthly_pnl_series(trades)
            if len(monthly_pnl) < 6:
                continue

            rows = []
            for ym, pnl in monthly_pnl:
                sig = signal_map.get(ym)
                if sig is None or sig["hurst"] is None or sig["adx"] is None or sig["er"] is None:
                    continue
                rows.append((ym, pnl, sig["hurst"], sig["adx"], sig["er"]))

            if len(rows) < 12:
                print(f"--- {key}：有效月份{len(rows)}个，样本不足，跳过 ---\n")
                continue

            pnls = np.array([r[1] for r in rows])
            hursts = np.array([r[2] for r in rows])
            adxs = np.array([r[3] for r in rows])
            ers = np.array([r[4] for r in rows])

            corr_h = np.corrcoef(hursts, pnls)[0, 1]
            corr_a = np.corrcoef(adxs, pnls)[0, 1]
            corr_e = np.corrcoef(ers, pnls)[0, 1]

            # 分组对比：指标低1/3 vs 高1/3 分组下，当月PnL均值/胜率
            def group_compare(signal_vals, label):
                order = np.argsort(signal_vals)
                n = len(order)
                low_idx = order[: n // 3]
                high_idx = order[-(n // 3):]
                low_pnl = pnls[low_idx]
                high_pnl = pnls[high_idx]
                low_win = (low_pnl > 0).mean() * 100
                high_win = (high_pnl > 0).mean() * 100
                print(f"    {label}：corr={np.corrcoef(signal_vals, pnls)[0,1]:+.3f}  "
                      f"低1/3组月均PnL={low_pnl.mean():+.1f} 正收益月占比{low_win:.0f}%  |  "
                      f"高1/3组月均PnL={high_pnl.mean():+.1f} 正收益月占比{high_win:.0f}%")

            print(f"--- {key}（{len(rows)}个有效月份，{rows[0][0]}~{rows[-1][0]}） ---")
            group_compare(hursts, "Hurst   ")
            group_compare(adxs, "ADX(14) ")
            group_compare(ers, "效率比ER")
            print()


if __name__ == "__main__":
    main()
