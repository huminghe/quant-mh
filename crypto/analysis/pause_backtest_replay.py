"""
历史触发事件的暂停 vs 不暂停事后回算

背景：2026-10-21调研确认"审查触发后是否暂停"无法做统计回测验证（原始条件——
盈亏比<1.58连续2月——全历史仅6次独立触发事件，样本太小太相关）。但6次具体
事件本身可以逐一回算"如果当时暂停了会怎样"——这不是统计检验，是描述性事后
复盘，不需要显著性，只摆出每次的实际结果。用固定资本分母（20000 USDT，对齐
项目回撤计算规范）。

2026-10-21第二轮：用户质疑"样本不够"的结论建立在单一严格阈值上，要求先
用 trigger_grid_scan.py 扫描更宽松的阈值组合。扫描结果：阈值放宽到2.00（仍
要求连续2个月）后，独立事件数从6次增至17次（`python trigger_grid_scan.py
--list-events --threshold 2.00 --consecutive 2` 产出）。本次改为跑这17次
扩展样本，EVENTS_ORIGINAL保留原6次口径用于对比，EVENTS_EXPANDED为新的17次。

2026-10-21第三轮：用户指出流程混乱——不该跳过"单指标基线回算"直接测
三指标联合判据。补齐胜率/Sharpe/MFE比值/CUSUM四个候选各自作为独立触发
判据时的暂停回算（EVENTS_WIN_RATE/EVENTS_SHARPE/EVENTS_MFE/EVENTS_CUSUM），
阈值选择见各常量上方注释，统一原则是"样本量跟盈亏比扩展口径的17次量级
可比，避免太少无回算意义、太多噪声过大"。

数据bug修复：当前日期2026-10-21，各策略最新数据只到10月中旬，10月是不完整
月份。trigger_grid_scan.py 按月滚动聚合时若不排除未走完的最新月份，会误判出
"2026-10-31"这类实际还未到达的触发月（已发现并在 trigger_grid_scan.py 里用
drop_incomplete_last_month 修复）。本文件的 EVENTS_WIN_RATE/EVENTS_MFE/
EVENTS_CUSUM 已用修复后的数据重新生成（分别从22→20次、18→17次，CUSUM的
4次里置换掉1条错误事件），EVENTS_SHARPE本身不含10月事件，不受影响。仍在
进行中的触发事件（窗口数据不足，无法回算"之后"表现）在回算时自动跳过。

口径：
  - 触发确认时点 = 连续2个月盈亏比<阈值的第2个月月末（下月初开始暂停）
  - 暂停版：暂停期间收益记0，之后恢复正常交易，与不暂停版在暂停结束时点对齐
  - 对比窗口：暂停期结束后 1/3/6 个月的累计收益（固定资本分母百分比）

用法：
  python pause_backtest_replay.py                    # 跑扩展样本（17次，盈亏比阈值2.00，默认）
  python pause_backtest_replay.py --metric original   # 原始样本（6次，盈亏比阈值1.58）
  python pause_backtest_replay.py --metric joint      # 双指标联合判据（盈亏比+胜率，7次）
  python pause_backtest_replay.py --metric win_rate   # 单指标基线：胜率
  python pause_backtest_replay.py --metric sharpe     # 单指标基线：滚动12月Sharpe
  python pause_backtest_replay.py --metric mfe        # 单指标基线：MFE比值
  python pause_backtest_replay.py --metric cusum      # 单指标基线：CUSUM变点检测
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import scan_files_auto
from rolling_pl_ratio import read_trade_pnls

CAPITAL = 20000.0  # 固定资本分母，对齐项目回撤计算规范

# 原始口径：盈亏比<1.58，连续2个月，6次独立触发事件
EVENTS_ORIGINAL = [
    ("BTC_v2", datetime.date(2022, 11, 30)),
    ("BTC_v2", datetime.date(2026, 6, 30)),  # 05-06月已满足连续2月，06月虽无交易仍算已确认
    ("DOGE_v2", datetime.date(2022, 4, 30)),
    ("DOGE_v2", datetime.date(2023, 5, 31)),
    ("ETH_v2", datetime.date(2023, 7, 31)),
    ("SOL_v2", datetime.date(2026, 6, 30)),
]

# 扩展口径：盈亏比<2.00（接近历史均值2.26，明显更宽松），连续2个月，17次独立触发事件
# 来源：trigger_grid_scan.py --list-events --threshold 2.00 --consecutive 2
EVENTS_EXPANDED = [
    ("BTC_v2", datetime.date(2022, 11, 30)),
    ("BTC_v2", datetime.date(2024, 7, 31)),
    ("BTC_v2", datetime.date(2025, 9, 30)),
    ("BTC_v2", datetime.date(2026, 7, 31)),
    ("DOGE_v2", datetime.date(2022, 4, 30)),
    ("DOGE_v2", datetime.date(2023, 3, 31)),
    ("DOGE_v2", datetime.date(2026, 1, 31)),
    ("DOGE_v2", datetime.date(2026, 4, 30)),
    ("DOGE_v3_205m", datetime.date(2022, 6, 30)),
    ("DOGE_v3_205m", datetime.date(2023, 6, 30)),
    ("ETH_v2", datetime.date(2023, 7, 31)),
    ("SOL_v2", datetime.date(2021, 8, 31)),
    ("SOL_v2", datetime.date(2022, 3, 31)),
    ("SOL_v2", datetime.date(2023, 6, 30)),
    ("SOL_v2", datetime.date(2024, 11, 30)),
    ("SOL_v2", datetime.date(2026, 5, 31)),
    ("SOL_v3_205m", datetime.date(2024, 8, 31)),
]

# 联合判据口径：盈亏比<2.00 且 胜率<历史均值*0.90，连续2个月，7次独立触发事件
# 验证目的不是扩大样本（联合判据本身样本更少），是检验"筛选质量是否比单指标更高"
# 来源：手动网格扫描（盈亏比+胜率联合条件），见 strategy_research_log.md 对应条目
EVENTS_JOINT = [
    ("BTC_v2", datetime.date(2026, 7, 31)),
    ("DOGE_v2", datetime.date(2022, 4, 30)),
    ("DOGE_v2", datetime.date(2023, 3, 31)),
    ("DOGE_v2", datetime.date(2026, 5, 31)),
    ("DOGE_v3_205m", datetime.date(2022, 6, 30)),
    ("DOGE_v3_205m", datetime.date(2023, 6, 30)),
    ("SOL_v2", datetime.date(2026, 5, 31)),
]

# 单指标基线：胜率<历史均值*0.70，连续2个月，20次
# 来源：trigger_grid_scan.py --metric win_rate（drop_incomplete_last_month修复后重新生成，
# 原22次里的2026-10-31两条是10月数据未走完被误判为触发月，修复后已剔除）
EVENTS_WIN_RATE = [
    ("BTC_v2", datetime.date(2020, 8, 31)),
    ("BTC_v2", datetime.date(2023, 9, 30)),
    ("BTC_v3_205m", datetime.date(2020, 8, 31)),
    ("BTC_v3_205m", datetime.date(2022, 9, 30)),
    ("BTC_v3_205m", datetime.date(2023, 8, 31)),
    ("BTC_v3_205m", datetime.date(2025, 3, 31)),
    ("DOGE_v3_205m", datetime.date(2020, 10, 31)),
    ("DOGE_v3_205m", datetime.date(2021, 11, 30)),
    ("DOGE_v3_205m", datetime.date(2022, 5, 31)),
    ("DOGE_v3_205m", datetime.date(2024, 2, 29)),
    ("DOGE_v3_205m", datetime.date(2024, 8, 31)),
    ("DOGE_v3_205m", datetime.date(2025, 5, 31)),
    ("DOGE_v3_205m", datetime.date(2026, 5, 31)),
    ("ETH_v3_205m", datetime.date(2023, 9, 30)),
    ("ETH_v3_205m", datetime.date(2024, 3, 31)),
    ("ETH_v3_205m", datetime.date(2026, 9, 30)),
    ("SOL_v3_205m", datetime.date(2024, 2, 29)),
    ("SOL_v3_205m", datetime.date(2025, 1, 31)),
    ("SOL_v3_205m", datetime.date(2026, 2, 28)),
    ("SOL_v3_205m", datetime.date(2026, 5, 31)),
]

# 单指标基线：滚动12月Sharpe<0，连续2个月，6次（口径对齐health_monitor.md规则表）
# 来源：trigger_grid_scan.py --metric sharpe
EVENTS_SHARPE = [
    ("BTC_v3_205m", datetime.date(2025, 4, 30)),
    ("DOGE_v2", datetime.date(2023, 1, 31)),
    ("DOGE_v2", datetime.date(2023, 10, 31)),
    ("DOGE_v2", datetime.date(2026, 8, 31)),
    ("DOGE_v3_205m", datetime.date(2022, 6, 30)),
    ("DOGE_v3_205m", datetime.date(2023, 12, 31)),
]

# 单指标基线：滚动3月MFE均值/全历史均值<0.70，连续2个月，17次
# 来源：trigger_grid_scan.py --metric mfe（drop_incomplete_last_month修复后重新生成，
# 原18次里SOL_v3_205m 2026-10-31这一条是10月数据未走完被误判为触发月，修复后已剔除）
EVENTS_MFE = [
    ("BTC_v2", datetime.date(2023, 9, 30)),
    ("BTC_v2", datetime.date(2025, 8, 31)),
    ("BTC_v2", datetime.date(2026, 7, 31)),
    ("BTC_v3_205m", datetime.date(2023, 8, 31)),
    ("BTC_v3_205m", datetime.date(2026, 6, 30)),
    ("DOGE_v2", datetime.date(2023, 10, 31)),
    ("DOGE_v2", datetime.date(2026, 6, 30)),
    ("DOGE_v3_205m", datetime.date(2020, 10, 31)),
    ("DOGE_v3_205m", datetime.date(2022, 1, 31)),
    ("DOGE_v3_205m", datetime.date(2023, 7, 31)),
    ("DOGE_v3_205m", datetime.date(2026, 6, 30)),
    ("ETH_v2", datetime.date(2023, 7, 31)),
    ("ETH_v2", datetime.date(2026, 7, 31)),
    ("ETH_v3_205m", datetime.date(2023, 6, 30)),
    ("ETH_v3_205m", datetime.date(2026, 7, 31)),
    ("SOL_v2", datetime.date(2026, 6, 30)),
    ("SOL_v3_205m", datetime.date(2026, 6, 30)),
]

# 单指标基线：CUSUM变点检测 k=0.5 h=3.0，4次
# 来源：trigger_grid_scan.py --list-cusum --cusum-k 0.5 --cusum-h 3.0（修复后重新生成，
# 原列表里ETH_v2 2026-10-31是10月数据未走完的误判报警，修复后替换为BTC_v3_205m 2025-4-30
# 这个真实报警，总数仍为4次）
EVENTS_CUSUM = [
    ("BTC_v2", datetime.date(2026, 9, 30)),
    ("BTC_v3_205m", datetime.date(2025, 4, 30)),
    ("DOGE_v2", datetime.date(2022, 4, 30)),
    ("DOGE_v2", datetime.date(2023, 7, 31)),
]

# 两两组合：盈亏比<2.00 且 Sharpe<0，连续1个月（连续2月仅1次太少，退一档取连续1月），5次
# 来源：trigger_grid_scan.py --list-joint2 pl_sharpe --threshold-a 2.00 --threshold-b 0.0 --consecutive 1
EVENTS_JOINT_PL_SHARPE = [
    ("DOGE_v2", datetime.date(2023, 2, 28)),
    ("DOGE_v2", datetime.date(2023, 7, 31)),
    ("DOGE_v2", datetime.date(2023, 11, 30)),
    ("DOGE_v3_205m", datetime.date(2022, 5, 31)),
    ("DOGE_v3_205m", datetime.date(2023, 11, 30)),
]

# 两两组合：盈亏比<2.00 且 MFE比值<0.70，连续2个月，6次
# 来源：trigger_grid_scan.py --list-joint2 pl_mfe --threshold-a 2.00 --threshold-b 0.70 --consecutive 2
EVENTS_JOINT_PL_MFE = [
    ("BTC_v2", datetime.date(2025, 9, 30)),
    ("BTC_v2", datetime.date(2026, 7, 31)),
    ("DOGE_v3_205m", datetime.date(2022, 6, 30)),
    ("DOGE_v3_205m", datetime.date(2023, 7, 31)),
    ("ETH_v2", datetime.date(2023, 7, 31)),
    ("SOL_v2", datetime.date(2026, 6, 30)),
]

# 两两组合：Sharpe<0 且 MFE比值<0.70，连续2个月，3次
# 来源：trigger_grid_scan.py --list-joint2 sharpe_mfe --threshold-a 0.0 --threshold-b 0.70 --consecutive 2
EVENTS_JOINT_SHARPE_MFE = [
    ("DOGE_v2", datetime.date(2023, 10, 31)),
    ("DOGE_v2", datetime.date(2026, 8, 31)),
    ("DOGE_v3_205m", datetime.date(2022, 6, 30)),
]

PAUSE_DAYS_OPTIONS = [30, 50]
FORWARD_WINDOWS_DAYS = {"1个月": 30, "3个月": 90, "6个月": 180}


def pct_return_in_window(trades: list[tuple[datetime.date, float]],
                          start: datetime.date, end: datetime.date) -> float:
    """区间内净损益总和 / 固定资本分母，单位百分比。"""
    total = sum(p for d, p in trades if start <= d <= end)
    return total / CAPITAL * 100


EVENT_SETS = {
    "original": (EVENTS_ORIGINAL, "原始口径（盈亏比<1.58，单指标）"),
    "expanded": (EVENTS_EXPANDED, "扩展口径（盈亏比<2.00，单指标，更宽松）"),
    "joint": (EVENTS_JOINT, "联合判据口径（盈亏比<2.00且胜率<历史均值*0.90）"),
    "win_rate": (EVENTS_WIN_RATE, "单指标基线（胜率<历史均值*0.70，连续2月）"),
    "sharpe": (EVENTS_SHARPE, "单指标基线（滚动12月Sharpe<0，连续2月）"),
    "mfe": (EVENTS_MFE, "单指标基线（MFE比值<0.70，连续2月）"),
    "cusum": (EVENTS_CUSUM, "单指标基线（CUSUM变点检测 k=0.5 h=3.0）"),
    "joint_pl_sharpe": (EVENTS_JOINT_PL_SHARPE, "两两组合（盈亏比<2.00且Sharpe<0，连续1月）"),
    "joint_pl_mfe": (EVENTS_JOINT_PL_MFE, "两两组合（盈亏比<2.00且MFE比值<0.70，连续2月）"),
    "joint_sharpe_mfe": (EVENTS_JOINT_SHARPE_MFE, "两两组合（Sharpe<0且MFE比值<0.70，连续2月）"),
}


def main():
    parser = argparse.ArgumentParser(description="历史触发事件暂停vs不暂停事后回算")
    parser.add_argument("--metric", choices=list(EVENT_SETS.keys()), default="expanded")
    args = parser.parse_args()
    events, label = EVENT_SETS[args.metric]

    files = scan_files_auto("~/Downloads/".replace("~", str(Path.home())))
    trades_cache = {}

    print(f"===== 历史触发事件：暂停 vs 不暂停 事后回算 —— {label}，共{len(events)}次 =====")
    print(f"固定资本分母：{CAPITAL} USDT，仅统计已平仓交易净损益\n")

    for key, trigger_date in events:
        if key not in trades_cache:
            trades_cache[key] = read_trade_pnls(files[key])
        trades = trades_cache[key]
        last_trade_date = max(d for d, _ in trades) if trades else None

        print(f"--- {key}  触发确认：{trigger_date} ---")
        if last_trade_date is not None and (last_trade_date - trigger_date).days < 30:
            print(f"  [跳过：触发点距最新数据仅{(last_trade_date - trigger_date).days}天，"
                  f"事件仍在进行中，无法回算\"之后\"表现]\n")
            continue

        for pause_days in PAUSE_DAYS_OPTIONS:
            pause_end = trigger_date + datetime.timedelta(days=pause_days)
            print(f"  [暂停{pause_days}天版，暂停至 {pause_end}]")

            for label, window_days in FORWARD_WINDOWS_DAYS.items():
                # 不暂停版：从触发点起，正常累计 window_days
                no_pause_end = trigger_date + datetime.timedelta(days=window_days)
                no_pause_ret = pct_return_in_window(trades, trigger_date + datetime.timedelta(days=1), no_pause_end)

                # 暂停版：暂停期收益记0，之后恢复正常交易，
                # 对比口径对齐到同一个"外部时间窗口"（trigger_date + window_days），
                # 暂停期内按0计，暂停结束后到窗口末正常计
                pause_version_end = trigger_date + datetime.timedelta(days=window_days)
                if pause_version_end <= pause_end:
                    pause_ret = 0.0  # 窗口还在暂停期内，全程0收益
                else:
                    pause_ret = pct_return_in_window(trades, pause_end + datetime.timedelta(days=1), pause_version_end)

                diff = pause_ret - no_pause_ret
                verdict = "暂停更好" if diff > 0 else ("不暂停更好" if diff < 0 else "打平")
                print(f"    {label}后：不暂停={no_pause_ret:+.2f}%  暂停{pause_days}天={pause_ret:+.2f}%  "
                      f"差值={diff:+.2f}%  ({verdict})")
        print()


if __name__ == "__main__":
    main()
