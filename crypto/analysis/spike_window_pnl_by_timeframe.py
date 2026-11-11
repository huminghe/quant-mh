"""
按策略时间框架拆分：门结构事件窗口内 vs 窗口外的逐笔盈亏对比。

背景：2026-11-11发现9个部署策略约51%近期亏损来自4次门结构事件（见
strategy_research_log.md同日条目），但该归因未区分策略时间框架。

⚠️ 2026-11-11更新：最初在8h框架用spike_bar_detector.py复测时，GK分位数法/
ATR倍数法命中率明显低于4h，一度被误判为"8h的K线收益率把4h尖峰抹平了"。
经核实这个判断错误——直接核查8h原始K线发现已知5次事件在8h上同样清晰可见
（如10-29 16:00那根8h bar收益率-4.63%），真实原因是检测阈值（GK 95分位/
ATR 3倍）在8h框架上偏松导致漏判，不是现象消失。该错误结论已撤回，不代表
"8h策略对4h级别冲击没有感知能力"，详见strategy_research_log.md同日条目。

本脚本不检测K线尖峰，直接用已知的门结构事件时间窗口（逐笔盈亏表已有数据），
按策略版本（v2=8h类 vs v3_205m/v5_3h=短周期类）拆分窗口内/窗口外盈亏，
对比两类策略受门结构事件影响的严重程度是否不同——这个问题本身与上面
"8h检测不到"的错误推论无关，两类策略客观上都承受了冲击，只是想看单笔
均亏幅度是否因持仓周期（bar周期越长、止损触发越晚）而不同。

用法：
  python spike_window_pnl_by_timeframe.py
  python spike_window_pnl_by_timeframe.py --dir ~/Downloads/
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rolling_pl_ratio import read_trade_pnls
from analysis_core import scan_files_auto

# 4次已知门结构事件所在日期（UTC），逐笔盈亏表按出场日期匹配，用事件日±1天
# 做窗口匹配避免同一事件因持仓跨日被漏记（见strategy_research_log.md 2026-11-11条目）
SPIKE_EVENT_DATES = [
    datetime.date(2026, 10, 29),
    datetime.date(2026, 11, 4),
    datetime.date(2026, 11, 5),
    datetime.date(2026, 11, 8),
    datetime.date(2026, 11, 10),
]
WINDOW_DAYS = 1  # 事件日前后各扩展1天


def in_spike_window(d: datetime.date) -> bool:
    for ev in SPIKE_EVENT_DATES:
        if abs((d - ev).days) <= WINDOW_DAYS:
            return True
    return False


def classify_timeframe(key: str) -> str:
    """按策略版本名归类时间框架类别。"""
    if key.endswith("_v3_205m"):
        return "短周期(205m)"
    if key.endswith("_v5_3h") or key.endswith("_3h"):
        return "短周期(3h)"
    if key.endswith("_v2") or key.endswith("_ema"):
        return "8h类(479m)"
    return "其他"


def summarize(trades: list[tuple[datetime.date, float]], since: datetime.date | None) -> dict:
    if since is not None:
        trades = [(d, p) for d, p in trades if d >= since]
    in_win = [p for d, p in trades if in_spike_window(d)]
    out_win = [p for d, p in trades if not in_spike_window(d)]
    return {
        "n_in": len(in_win), "pnl_in": sum(in_win),
        "n_out": len(out_win), "pnl_out": sum(out_win),
    }


def main():
    ap = argparse.ArgumentParser(description="按策略时间框架拆分门结构窗口内外盈亏")
    ap.add_argument("--dir", default=os.path.expanduser("~/Downloads/"))
    ap.add_argument("--since", default="2026-10-01",
                    help="只统计该日期以来的交易（默认2026-10-01，对齐strategy_research_log.md 11-11归因口径），传空字符串表示不限制")
    args = ap.parse_args()
    since = datetime.date.fromisoformat(args.since) if args.since else None

    files = scan_files_auto(args.dir)
    if not files:
        print(f"在 {args.dir} 未找到策略 xlsx 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：{sorted(files.keys())}")
    print(f"统计范围：{since or '全历史'} 以来")
    print(f"门结构事件窗口：{[str(d) for d in SPIKE_EVENT_DATES]} 各±{WINDOW_DAYS}天\n")

    by_class: dict[str, dict] = {}
    print(f"{'策略':<20} {'分类':<12} {'窗口内笔数':>8} {'窗口内盈亏':>12} {'窗口外笔数':>8} {'窗口外盈亏':>12}")
    for key, fp in sorted(files.items()):
        trades = read_trade_pnls(fp)
        s = summarize(trades, since)
        cls = classify_timeframe(key)
        print(f"{key:<20} {cls:<12} {s['n_in']:>10} {s['pnl_in']:>12.1f} {s['n_out']:>10} {s['pnl_out']:>12.1f}")

        agg = by_class.setdefault(cls, {"n_in": 0, "pnl_in": 0.0, "n_out": 0, "pnl_out": 0.0})
        agg["n_in"] += s["n_in"]
        agg["pnl_in"] += s["pnl_in"]
        agg["n_out"] += s["n_out"]
        agg["pnl_out"] += s["pnl_out"]

    print(f"\n{'===== 按时间框架分类汇总 ====='}")
    print(f"{'分类':<12} {'窗口内笔数':>8} {'窗口内盈亏':>12} {'窗口内单笔均值':>14} "
          f"{'窗口外笔数':>8} {'窗口外盈亏':>12} {'窗口外单笔均值':>14}")
    for cls, agg in sorted(by_class.items()):
        avg_in = agg["pnl_in"] / agg["n_in"] if agg["n_in"] else 0.0
        avg_out = agg["pnl_out"] / agg["n_out"] if agg["n_out"] else 0.0
        print(f"{cls:<12} {agg['n_in']:>10} {agg['pnl_in']:>12.1f} {avg_in:>14.1f} "
              f"{agg['n_out']:>10} {agg['pnl_out']:>12.1f} {avg_out:>14.1f}")


if __name__ == "__main__":
    main()
