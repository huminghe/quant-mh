"""
滚动3个月盈亏比/胜率计算

按月汇总已平仓交易的盈利/亏损，计算滚动3个月窗口的盈亏比和胜率，
对照 health_monitor.md 的监控指标（滚动3个月盈亏比基准2.26/预警线1.58，
滚动3个月胜率基准37.5%/预警线30%）。

补充说明：run_all.py 生成的结论 MD 只有全周期汇总和年度分解，没有滚动3个月
这个维度；health_report.py 的 --latest 模式读的就是那份 MD，所以给不出真实
的滚动盈亏比。本脚本单独计算，结果需要手动写入研究日志或传给
health_report.py --rr/--wr。

用法：
  python rolling_pl_ratio.py                  # 扫描 ~/Downloads 最新 xlsx
  python rolling_pl_ratio.py --dir ~/Downloads/ --window 3
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import openpyxl
from analysis_utils import _get_trade_sheet
from analysis_core import scan_files_auto

WINDOW_MONTHS = 3
PL_RATIO_WARN = 1.58
# 胜率预警线按策略类型区分：v3_205m（长周期）历史均值胜率显著低于v2（短周期），
# 统一用一条阈值会让v3_205m系列常年"破线"、阈值形同虚设（2026-08-14 发现并修正）
WIN_RATE_WARN_V2 = 27.0
WIN_RATE_WARN_V3_205M = 22.0


def win_rate_warn_threshold(key: str) -> float:
    """按策略 key（如 BTC_v2/BTC_v3_205m）返回对应的胜率预警阈值。"""
    if key.endswith("_v3_205m"):
        return WIN_RATE_WARN_V3_205M
    return WIN_RATE_WARN_V2


def read_trade_pnls(filepath: str) -> list[tuple[datetime.date, float]]:
    """读取已平仓交易的 (日期, 净损益USDT) 列表，只取出场行避免重复计数。"""
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = _get_trade_sheet(wb)
    result = []
    for r in ws.iter_rows(values_only=True):
        t = str(r[1] or "")
        if "出场" not in t:
            continue
        dt_val = r[2]
        if not isinstance(dt_val, datetime.datetime):
            continue
        profit = r[7]
        if profit is None:
            continue
        result.append((dt_val.date(), float(profit)))
    wb.close()
    return sorted(result)


def monthly_win_loss(trades: list[tuple[datetime.date, float]]) -> dict[tuple[int, int], dict]:
    """按月汇总胜/负交易的总盈利、总亏损、笔数。"""
    monthly: dict[tuple[int, int], dict] = {}
    for d, p in trades:
        ym = (d.year, d.month)
        m = monthly.setdefault(ym, {"win_sum": 0.0, "win_n": 0, "loss_sum": 0.0, "loss_n": 0})
        if p > 0:
            m["win_sum"] += p
            m["win_n"] += 1
        elif p < 0:
            m["loss_sum"] += -p
            m["loss_n"] += 1
    return monthly


def rolling_pl_ratio(monthly: dict[tuple[int, int], dict],
                     window: int = WINDOW_MONTHS) -> list[dict]:
    """计算滚动 window 个月的盈亏比和胜率。"""
    months = sorted(monthly.keys())
    results = []
    for i in range(window - 1, len(months)):
        wm = months[i - window + 1: i + 1]
        win_sum = sum(monthly[m]["win_sum"] for m in wm)
        win_n = sum(monthly[m]["win_n"] for m in wm)
        loss_sum = sum(monthly[m]["loss_sum"] for m in wm)
        loss_n = sum(monthly[m]["loss_n"] for m in wm)
        total_n = win_n + loss_n
        if total_n == 0:
            continue
        avg_win = win_sum / win_n if win_n else 0.0
        avg_loss = loss_sum / loss_n if loss_n else 0.0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
        win_rate = win_n / total_n * 100
        results.append({
            "end_month": wm[-1],
            "pl_ratio": pl_ratio,
            "win_rate": win_rate,
            "n_trades": total_n,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="滚动3个月盈亏比/胜率计算")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--window", type=int, default=WINDOW_MONTHS,
                        help="滚动窗口月数（默认3）")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="窗口内样本数低于该值标记为'不作数'（默认10）")
    args = parser.parse_args()

    files = scan_files_auto(args.dir)
    if not files:
        print(f"在 {args.dir} 未找到策略 xlsx 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：{sorted(files.keys())}\n")

    # 合并所有策略的交易，计算整体滚动盈亏比/胜率（对齐 health_monitor.md 的口径）
    all_trades: list[tuple[datetime.date, float]] = []
    per_strategy: dict[str, list[dict]] = {}

    for key, fp in files.items():
        trades = read_trade_pnls(fp)
        all_trades.extend(trades)
        monthly = monthly_win_loss(trades)
        rows = rolling_pl_ratio(monthly, args.window)
        per_strategy[key] = rows

    combined_monthly = monthly_win_loss(all_trades)
    combined_rows = rolling_pl_ratio(combined_monthly, args.window)

    print(f"===== 合并（全部 {len(files)} 个策略）滚动 {args.window} 个月盈亏比/胜率 =====")
    print("注：合并结果混合了v2/v3_205m两类策略，胜率无统一阈值可比，仅盈亏比预警有效")
    print(f"{'月份':<10} {'盈亏比':>8} {'胜率%':>8} {'样本数':>8} {'状态':<10}")
    for row in combined_rows:
        ym = row["end_month"]
        month_str = f"{ym[0]}-{ym[1]:02d}"
        note = ""
        if row["n_trades"] < args.min_trades:
            note = "样本不足，不作数"
        elif row["pl_ratio"] < PL_RATIO_WARN:
            note = "破盈亏比预警线"
        print(f"{month_str:<10} {row['pl_ratio']:>8.2f} {row['win_rate']:>7.1f}% "
              f"{row['n_trades']:>8} {note:<10}")

    print(f"\n预警线：盈亏比 < {PL_RATIO_WARN}")

    print(f"\n===== 各策略最新窗口（胜率阈值按策略类型区分：v2<{WIN_RATE_WARN_V2}%，v3_205m<{WIN_RATE_WARN_V3_205M}%）=====")
    watchlist = []
    for key, rows in sorted(per_strategy.items()):
        if not rows:
            print(f"{key}: 无足够数据")
            continue
        last = rows[-1]
        ym = last["end_month"]
        wr_breach = last["win_rate"] < win_rate_warn_threshold(key)
        pl_breach = last["pl_ratio"] < PL_RATIO_WARN
        wr_note = " <-- 破胜率预警线" if wr_breach else ""
        pl_note = " <-- 破盈亏比预警线" if pl_breach else ""
        print(f"{key}: {ym[0]}-{ym[1]:02d} 盈亏比={last['pl_ratio']:.2f}{pl_note} "
              f"胜率={last['win_rate']:.1f}%{wr_note} 样本={last['n_trades']}")
        if wr_breach:
            watchlist.append((key, last["pl_ratio"], pl_breach))

    # 胜率观察名单：胜率不单独触发审查，但需要每月追踪，见 health_monitor.md「胜率观察名单机制」
    print("\n===== 胜率观察名单（本月破线，需下月复核盈亏比是否同步走弱）=====")
    if not watchlist:
        print("本月无策略胜率破线")
    else:
        for key, pl_ratio, pl_breach in watchlist:
            status = "盈亏比已同步破线，联合判据满足，应提前介入归因" if pl_breach else f"盈亏比={pl_ratio:.2f}健康，判断为胜率端正常波动"
            print(f"{key}: {status}")


if __name__ == "__main__":
    main()
