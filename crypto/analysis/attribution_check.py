"""
审查触发后归因客观化检查

替代 health_monitor.md 审查流程里偏主观的人工读图判断（"是否有异常大亏损"
"平均持仓是否骤降"），输出四项可复算的客观指标，供审查时对照，不自动下结论。

背景：2026-10-21 调研确认"审查触发后是否暂停"这个决策样本量太小（全历史仅6次
独立触发事件，且高度相关），无法做统计回测验证（见 strategy_research_log.md）。
转而把归因阶段的三步人工判断客观化，降低主观判断占比，不追求统计显著性。

四个维度：
  1. 异常大亏损剔除法：单笔亏损 > 全周期均值亏损 1.5倍 的笔数及剔除后盈亏比
  2. 持仓时长对比：近N笔平均持仓K线 vs 全周期均值的比值
  3. MFE分布对比：近N笔有利波动%中位数 vs 全周期中位数（区分"趋势变短"还是"假突破增多"）
  4. 跨策略同期性：同一批文件里其他策略是否在同期也触发审查（系统性 vs 个体退化）

用法：
  python attribution_check.py --dir ~/Downloads/ --key BTC_ema --recent 30
  python attribution_check.py --dir ~/Downloads/ --key BTC_ema --recent 30 --months 2026-09,2026-10
"""
import argparse
import datetime
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import openpyxl
from analysis_utils import _get_trade_sheet
from analysis_core import scan_files_auto
from rolling_pl_ratio import (
    PL_RATIO_WARN, monthly_win_loss, rolling_pl_ratio, win_rate_warn_threshold,
)

LARGE_LOSS_MULTIPLIER = 1.5  # 异常大亏损判定：单笔亏损 > 全周期均值亏损 的倍数


def read_trade_detail(filepath: str) -> list[dict]:
    """读取已平仓交易明细：日期、净损益、有利波动%、持续K线数。"""
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
        result.append({
            "date": dt_val.date(),
            "pnl": float(profit),
            "mfe_pct": float(r[11]) if r[11] is not None else None,
            "bars": float(r[16]) if r[16] is not None else None,
        })
    wb.close()
    return sorted(result, key=lambda x: x["date"])


def check_large_loss(trades: list[dict], recent_n: int) -> dict:
    """维度1：异常大亏损剔除法。
    阈值用全历史亏损分布标定（避免近期样本量小导致阈值不稳），
    但只检查「近期窗口」内是否出现超出历史分布的极端亏损——
    消息面冲击是近期事件，不该用全历史范围的亏损笔数来判断。
    """
    all_losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    if not all_losses:
        return {"note": "无亏损交易，跳过"}
    avg_loss = abs(sum(all_losses) / len(all_losses))
    threshold = avg_loss * LARGE_LOSS_MULTIPLIER

    recent = trades[-recent_n:]
    large_recent = [t for t in recent if t["pnl"] < -threshold]

    r_losses = [t["pnl"] for t in recent if t["pnl"] < 0]
    r_wins = [t["pnl"] for t in recent if t["pnl"] > 0]
    pl_ratio_recent_raw = (
        (sum(r_wins) / len(r_wins)) / (abs(sum(r_losses)) / len(r_losses))
        if r_wins and r_losses else None
    )

    filtered = [t for t in recent if t not in large_recent]
    f_losses = [t["pnl"] for t in filtered if t["pnl"] < 0]
    f_wins = [t["pnl"] for t in filtered if t["pnl"] > 0]
    pl_ratio_filtered = None
    if f_wins and f_losses:
        pl_ratio_filtered = (sum(f_wins) / len(f_wins)) / (abs(sum(f_losses)) / len(f_losses))

    return {
        "avg_loss_all_history": round(avg_loss, 2),
        "large_loss_threshold": round(threshold, 2),
        "recent_n": len(recent),
        "large_loss_count_in_recent": len(large_recent),
        "large_loss_dates": [t["date"].isoformat() for t in large_recent],
        "pl_ratio_recent_raw": round(pl_ratio_recent_raw, 3) if pl_ratio_recent_raw else None,
        "pl_ratio_recent_after_exclude": round(pl_ratio_filtered, 3) if pl_ratio_filtered else None,
        "verdict": (
            "近期存在异常大亏损，剔除后盈亏比显著回升，倾向消息面冲击"
            if large_recent and pl_ratio_filtered and pl_ratio_recent_raw
            and pl_ratio_filtered > pl_ratio_recent_raw * 1.2
            else "近期无显著异常大亏损或剔除后无明显改善，不支持消息面冲击归因"
        ),
    }


def check_holding_period(trades: list[dict], recent_n: int) -> dict:
    """维度2：持仓时长对比（横盘磨损的直接代理）。"""
    bars_all = [t["bars"] for t in trades if t["bars"] is not None]
    if not bars_all:
        return {"note": "无持续K线数据，跳过"}
    recent = bars_all[-recent_n:]
    avg_all = statistics.mean(bars_all)
    avg_recent = statistics.mean(recent)
    ratio = avg_recent / avg_all if avg_all else None
    return {
        "avg_bars_all_history": round(avg_all, 2),
        "avg_bars_recent_n": round(avg_recent, 2),
        "recent_n": len(recent),
        "ratio": round(ratio, 3) if ratio else None,
        "verdict": (
            "近期持仓时长显著低于历史均值，支持横盘磨损/假突破增多"
            if ratio and ratio < 0.7
            else "持仓时长未显著偏离历史均值，不支持横盘磨损归因"
        ),
    }


def check_mfe_distribution(trades: list[dict], recent_n: int) -> dict:
    """维度3：MFE（有利波动%）分布对比。
    MFE收窄但胜率不变 -> 趋势耗尽（波段变短，方向仍对）
    MFE也下降且方差增大 -> 假突破增多（入场质量下降）
    """
    mfe_all = [t["mfe_pct"] for t in trades if t["mfe_pct"] is not None]
    if not mfe_all:
        return {"note": "无有利波动数据，跳过"}
    recent = mfe_all[-recent_n:]
    median_all = statistics.median(mfe_all)
    median_recent = statistics.median(recent)
    ratio = median_recent / median_all if median_all else None
    return {
        "median_mfe_all_history": round(median_all, 3),
        "median_mfe_recent_n": round(median_recent, 3),
        "recent_n": len(recent),
        "ratio": round(ratio, 3) if ratio else None,
        "verdict": (
            "近期MFE中位数显著收窄，支持趋势耗尽/假突破增多"
            if ratio and ratio < 0.7
            else "MFE分布未显著偏离历史，不支持趋势耗尽归因"
        ),
    }


def check_concurrent_trigger(all_strategies: dict, months: list[tuple[int, int]]) -> dict:
    """维度4：跨策略同期性检查。
    检查同一批文件里其他策略是否在同一时间窗口也触发审查（盈亏比<PL_RATIO_WARN）。
    多个独立策略同期触发，倾向系统性市场环境问题；仅单一策略触发，倾向个体退化。
    """
    triggered_others = []
    for key, trades in all_strategies.items():
        monthly = monthly_win_loss(trades)
        rows = rolling_pl_ratio(monthly)
        row_by_month = {r["end_month"]: r for r in rows}
        breach_months = [m for m in months if m in row_by_month and row_by_month[m]["pl_ratio"] < PL_RATIO_WARN]
        if breach_months:
            triggered_others.append({
                "key": key,
                "breach_months": [f"{y}-{m:02d}" for y, m in breach_months],
            })
    return {
        "checked_months": [f"{y}-{m:02d}" for y, m in months],
        "co_triggered_strategies": triggered_others,
        "co_triggered_count": len(triggered_others),
        "verdict": (
            f"同期另有 {len(triggered_others)} 个策略触发，倾向系统性市场环境问题，"
            "不宜仅对单一策略做参数层面调整"
            if len(triggered_others) >= 2
            else "同期无/仅1个其他策略触发，倾向个体策略退化，可聚焦单策略归因"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="审查触发后归因客观化检查")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--key", required=True,
                        help="目标策略 key，如 BTC_ema / BTC_v2")
    parser.add_argument("--recent", type=int, default=30,
                        help="「近期」窗口的笔数（默认30笔）")
    parser.add_argument("--months", default=None,
                        help="跨策略同期性检查覆盖的月份，逗号分隔，如 2026-09,2026-10；"
                             "默认取该策略最近2个月")
    args = parser.parse_args()

    files = scan_files_auto(args.dir)
    if args.key not in files:
        print(f"未找到策略 {args.key}，可用：{sorted(files.keys())}")
        sys.exit(1)

    trades = read_trade_detail(files[args.key])
    if not trades:
        print(f"{args.key} 无有效交易记录")
        sys.exit(1)

    print(f"===== {args.key} 归因客观化检查（共 {len(trades)} 笔已平仓交易）=====\n")

    print(f"【维度1：消息面冲击 —— 异常大亏损剔除法（近{args.recent}笔）】")
    r1 = check_large_loss(trades, args.recent)
    for k, v in r1.items():
        print(f"  {k}: {v}")

    print(f"\n【维度2：横盘磨损 —— 持仓时长对比（近{args.recent}笔）】")
    r2 = check_holding_period(trades, args.recent)
    for k, v in r2.items():
        print(f"  {k}: {v}")

    print(f"\n【维度3：趋势耗尽 —— MFE有利波动分布对比（近{args.recent}笔）】")
    r3 = check_mfe_distribution(trades, args.recent)
    for k, v in r3.items():
        print(f"  {k}: {v}")

    if args.months:
        months = []
        for part in args.months.split(","):
            y, m = part.strip().split("-")
            months.append((int(y), int(m)))
    else:
        last_date = trades[-1]["date"]
        months = [(last_date.year, last_date.month)]
        prev = last_date.month - 1 or 12
        prev_year = last_date.year if last_date.month > 1 else last_date.year - 1
        months.insert(0, (prev_year, prev))

    print(f"\n【维度4：跨策略同期性检查】")
    all_trades_by_key = {k: read_trade_detail(fp) for k, fp in files.items()}
    # 转换为 rolling_pl_ratio 需要的 (date, pnl) 元组列表
    all_trades_tuples = {k: [(t["date"], t["pnl"]) for t in v] for k, v in all_trades_by_key.items()}
    r4 = check_concurrent_trigger(all_trades_tuples, months)
    for k, v in r4.items():
        print(f"  {k}: {v}")

    print("\n===== 四维度结论汇总（供审查时参考，不自动决策）=====")
    print(f"  1. {r1.get('verdict', 'N/A')}")
    print(f"  2. {r2.get('verdict', 'N/A')}")
    print(f"  3. {r3.get('verdict', 'N/A')}")
    print(f"  4. {r4.get('verdict', 'N/A')}")


if __name__ == "__main__":
    main()
