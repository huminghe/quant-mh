"""
资金费率异常触发仓位动作：历史回放验证

背景：funding_rate_monitor.py 检测出的"持续偏离"事件只是市场结构异常信号，
本身不构成降风险的证据。用户要求验证的是能否真的用到策略里降低风险——
不是"异常检测准不准"，是"触发后采取仓位动作，历史回放Sharpe/回撤是否真的
改善"。本脚本做这一步验证。

规则设计：
  - 用 funding_rate_monitor.py 同样的z-score方法，对每个标的算出每日是否
    处于"持续偏离"状态（连续3期=1天内，z-score同向超过阈值）
  - 触发日的次日起，该标的仓位打折 FACTOR（默认0.5），持续 HOLD_DAYS 天
    （默认3天），之后恢复满仓，除非又新触发
  - 用 T-1 及更早的信号决定 T 日仓位（避免未来数据）：交易出场日期是T，
    用T-1日及之前已确定的"是否处于折扣期"状态
  - 对比：原始版 vs 折扣版的全历史累计收益、月度Sharpe、最大回撤
    （口径与 trend_scaling_replay.py 一致：固定资本分母，月度PnL算Sharpe不年化）

注意：这是对"资金费率异常"这个信号本身的降仓验证，不是对"山寨币vs主流币"
分类假设的验证——哪个标的触发次数多/少不重要，只看规则是否真的降风险。

用法：
  python funding_rate_action_replay.py --dir ~/Downloads/
  python funding_rate_action_replay.py --dir ~/Downloads/ --factor 0.3 --hold-days 5
"""
import argparse
import datetime
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from analysis_core import scan_files_auto
from rolling_pl_ratio import read_trade_pnls
from trigger_grid_scan import DEPLOYED_SUFFIXES
from funding_rate_monitor import ASSETS, SYMBOL_MAP, fetch_funding_history, Z_THRESHOLD, PERSIST_PERIODS, PERSIST_Z_THRESHOLD

CAPITAL = 20000.0
FUNDING_LOOKBACK_DAYS = 730  # 拉够长历史覆盖所有策略交易期


def build_discount_calendar(asset: str, factor: float, hold_days: int) -> dict:
    """返回 {date: 仓位折扣系数} 的日度日历，覆盖该标的全部触发事件及其持续折扣期。"""
    symbol = SYMBOL_MAP[asset]
    print(f"  拉取 {symbol} 资金费率历史（近{FUNDING_LOOKBACK_DAYS}天）...", end=" ", flush=True)
    rates = fetch_funding_history(symbol, FUNDING_LOOKBACK_DAYS)
    print(f"done ({len(rates)}期)")

    values = np.array([r["fundingRate"] for r in rates])
    # Binance返回UTC时间，TV交易记录的date是UTC+8 naive datetime直接取.date()，
    # 这里必须先+8小时再取date，否则日界附近（UTC+8的00:00-08:00）会与交易记录错位一天
    dates = [(datetime.datetime.fromisoformat(r["datetime"].replace("Z", "+00:00"))
              + datetime.timedelta(hours=8)).date() for r in rates]

    mean, std = values.mean(), values.std(ddof=1)
    z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)

    # 找持续偏离事件的触发日（事件窗口的最后一期所在日，即确认触发的那一天）
    trigger_dates = set()
    i = 0
    while i < len(z_scores) - PERSIST_PERIODS + 1:
        window = z_scores[i:i + PERSIST_PERIODS]
        if np.all(window > PERSIST_Z_THRESHOLD) or np.all(window < -PERSIST_Z_THRESHOLD):
            trigger_dates.add(dates[i + PERSIST_PERIODS - 1])
            i += PERSIST_PERIODS
        else:
            i += 1

    # 展开成日度折扣日历：触发日次日起 hold_days 天打折
    calendar = {}
    for td in trigger_dates:
        for offset in range(1, hold_days + 1):
            calendar[td + datetime.timedelta(days=offset)] = factor
    return calendar


def sharpe_from_monthly(monthly_pnls: list[float]) -> float | None:
    if len(monthly_pnls) < 2:
        return None
    std = np.std(monthly_pnls, ddof=1)
    return float(np.mean(monthly_pnls) / std) if std > 0 else None


def max_drawdown(monthly_pnls: list[float], capital: float) -> float:
    cum = np.cumsum(monthly_pnls) / capital * 100
    peak = np.maximum.accumulate(cum)
    return (cum - peak).min() if len(cum) else 0.0


def monthly_agg(trades: list[tuple], calendar: dict, factor_default: float = 1.0) -> dict:
    monthly = {}
    for d, p in trades:
        ym = (d.year, d.month)
        f = calendar.get(d, factor_default)
        monthly.setdefault(ym, 0.0)
        monthly[ym] += p * f
    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/Downloads/")
    parser.add_argument("--factor", type=float, default=0.5, help="触发后仓位折扣系数")
    parser.add_argument("--hold-days", type=int, default=3, help="折扣持续天数")
    args = parser.parse_args()
    directory = args.dir.replace("~", str(Path.home()))

    files = scan_files_auto(directory)
    deployed = {k: v for k, v in files.items() if k.endswith(DEPLOYED_SUFFIXES)}

    print(f"===== 资金费率异常触发仓位动作回放（折扣={args.factor}，"
          f"持续{args.hold_days}天） =====\n")

    calendars = {}
    for asset in ASSETS:
        calendars[asset] = build_discount_calendar(asset, args.factor, args.hold_days)
        print(f"    {asset}：{len(set(calendars[asset].values()) and calendars[asset])}个折扣日历条目\n")

    summary_rows = []
    for asset in ASSETS:
        calendar = calendars[asset]
        strat_keys = sorted(k for k in deployed if k.startswith(asset + "_"))
        for key in strat_keys:
            trades = read_trade_pnls(deployed[key])
            if not trades:
                continue

            orig_monthly = monthly_agg(trades, {})
            scaled_monthly = monthly_agg(trades, calendar)

            months_sorted = sorted(set(orig_monthly) | set(scaled_monthly))
            if len(months_sorted) < 12:
                continue

            orig_vals = [orig_monthly.get(ym, 0.0) for ym in months_sorted]
            scaled_vals = [scaled_monthly.get(ym, 0.0) for ym in months_sorted]

            orig_total = sum(orig_vals) / CAPITAL * 100
            scaled_total = sum(scaled_vals) / CAPITAL * 100
            orig_sharpe = sharpe_from_monthly(orig_vals)
            scaled_sharpe = sharpe_from_monthly(scaled_vals)
            orig_mdd = max_drawdown(orig_vals, CAPITAL)
            scaled_mdd = max_drawdown(scaled_vals, CAPITAL)

            n_discount_days = sum(1 for d, _ in trades if d in calendar)

            print(f"--- {key}（{len(months_sorted)}个月，触发折扣的交易日{n_discount_days}天次） ---")
            print(f"    原始：  累计收益={orig_total:+.1f}%  Sharpe={orig_sharpe if orig_sharpe is None else f'{orig_sharpe:.3f}'}  最大回撤={orig_mdd:.1f}%")
            print(f"    触发后：累计收益={scaled_total:+.1f}%  Sharpe={scaled_sharpe if scaled_sharpe is None else f'{scaled_sharpe:.3f}'}  最大回撤={scaled_mdd:.1f}%")
            sharpe_diff = (scaled_sharpe - orig_sharpe) if (orig_sharpe is not None and scaled_sharpe is not None) else None
            print(f"    差值：  收益{scaled_total - orig_total:+.1f}pp  Sharpe{'N/A' if sharpe_diff is None else f'{sharpe_diff:+.3f}'}  回撤改善{scaled_mdd - orig_mdd:+.1f}pp\n")

            summary_rows.append({
                "key": key, "orig_sharpe": orig_sharpe, "scaled_sharpe": scaled_sharpe,
                "orig_mdd": orig_mdd, "scaled_mdd": scaled_mdd,
                "orig_total": orig_total, "scaled_total": scaled_total,
            })

    print("===== 汇总 =====")
    n = len(summary_rows)
    sharpe_improved = sum(1 for r in summary_rows if r["orig_sharpe"] and r["scaled_sharpe"] and r["scaled_sharpe"] > r["orig_sharpe"])
    mdd_improved = sum(1 for r in summary_rows if r["scaled_mdd"] > r["orig_mdd"])
    total_improved = sum(1 for r in summary_rows if r["scaled_total"] > r["orig_total"])
    print(f"策略数={n}  Sharpe改善={sharpe_improved}/{n}  最大回撤改善={mdd_improved}/{n}  累计收益改善={total_improved}/{n}")


if __name__ == "__main__":
    main()
