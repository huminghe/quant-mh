"""
触发条件网格扫描：多阈值 x 多窗口 x 多指标，统计各组合能筛出多少独立触发事件

背景：2026-10-21 调研认为"连续2个月盈亏比<1.58"样本太少（全历史仅6次独立事件）
无法做统计回测验证。但这个结论建立在单一阈值+单一指标上，未扫描其他组合就下
结论是不充分的。本脚本把阈值、连续月数、滚动窗口月数、指标种类都做成可配置
维度，扫描出每个组合的独立触发事件数，用于判断样本量能否通过放宽条件做大。

独立触发事件定义：连续破线段的起点算一次（同一段内后续月份不重复计数），
跟 health_monitor.md「连续2个月触发审查」的口径一致，只是把"2个月"和
"1.58"都变成扫描变量。

指标支持：
  - pl_ratio：滚动3个月盈亏比（原始指标）
  - win_rate：滚动3个月胜率（按策略类型 v2/v3_205m 分别用相对历史均值的比例阈值）
  - sharpe：滚动12个月Sharpe（月度净盈亏均值/标准差，不年化，口径对齐 health_monitor.md
    2026-08-14"滚动Sharpe纳入"条目）
  - mfe_ratio：滚动3个月MFE（有利波动%）均值 vs 全历史均值的比值，口径对齐
    attribution_check.py 维度3，但这里按月滚动而非"近N笔"，以便跟其他月度指标联合判据
  - cusum：月度净盈亏序列的标准CUSUM变点检测（检测均值下移），历史日志提过但从未
    实现验证的候选，性质与前四个不同——是"检测持续性漂移"而非"原始值比阈值"

2026-10-21第二轮：新增 scan_joint_3metric，测试"盈亏比+Sharpe+MFE比值"三指标联合判据。
2026-10-21第三轮：用户指出流程混乱（跳过了单指标基线回算直接测三指标组合），
重新按"单指标回算→两两组合→三三组合"的顺序补齐，CUSUM作为第5个候选先补实现。

用法：
  python trigger_grid_scan.py --dir ~/Downloads/
  python trigger_grid_scan.py --dir ~/Downloads/ --metric win_rate
  python trigger_grid_scan.py --dir ~/Downloads/ --metric joint3
  python trigger_grid_scan.py --dir ~/Downloads/ --metric cusum
"""
import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import scan_files_auto
from attribution_check import read_trade_detail
from rolling_pl_ratio import monthly_win_loss, read_trade_pnls, rolling_pl_ratio

# 实盘部署范围：v2 + v3_205m 系列，口径对齐 health_monitor.md
DEPLOYED_SUFFIXES = ("_v2", "_v3_205m")

PL_RATIO_THRESHOLDS = [1.30, 1.40, 1.58, 1.70, 1.80, 2.00]
WIN_RATE_REL_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]  # 相对该策略历史均值胜率的比例
CONSECUTIVE_MONTHS_OPTIONS = [1, 2, 3]
ROLLING_WINDOW_OPTIONS = [3]  # health_monitor.md 固定用3个月滚动窗口，不展开扫描
SHARPE_WINDOW_MONTHS = 12  # 滚动Sharpe窗口，对齐 health_monitor.md
MFE_RATIO_REL_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]  # 滚动3月MFE均值 / 全历史均值


def drop_incomplete_last_month(monthly_items: list, last_trade_date) -> list:
    """丢弃最后一个月（若该月月末日期晚于最新交易日期，说明当月数据还没走完）。

    rolling_pl_ratio.py 的盈亏比/胜率计算因为历史上运行时数据刚好跑到月末附近，
    未必会遇到这个问题，但本轮新增的Sharpe/MFE/CUSUM扫描是在月中（10月19-21日）
    运行，10月只有约2/3的数据，若不截断会把"不完整月"误判为真实的触发月份。
    """
    import calendar
    if not monthly_items:
        return monthly_items
    last_ym = monthly_items[-1][0]
    y, m = last_ym
    last_day = calendar.monthrange(y, m)[1]
    if last_trade_date.day < last_day or last_trade_date.month != m or last_trade_date.year != y:
        return monthly_items[:-1]
    return monthly_items


def monthly_pnl_series(trades: list[tuple]) -> list[tuple]:
    """按月汇总净盈亏总和，返回[(年月, 月度净盈亏总和), ...]，用于算滚动Sharpe。"""
    monthly: dict[tuple, float] = {}
    for d, p in trades:
        ym = (d.year, d.month)
        monthly[ym] = monthly.get(ym, 0.0) + p
    result = sorted(monthly.items())
    if trades:
        result = drop_incomplete_last_month(result, max(d for d, _ in trades))
    return result


def rolling_sharpe(monthly_pnl: list[tuple], window: int = SHARPE_WINDOW_MONTHS) -> list[dict]:
    """滚动window个月Sharpe = 月度净盈亏均值/标准差（不年化）。"""
    results = []
    for i in range(window - 1, len(monthly_pnl)):
        wm = monthly_pnl[i - window + 1: i + 1]
        pnls = [p for _, p in wm]
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0.0
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else None
        results.append({"end_month": wm[-1][0], "sharpe": sharpe})
    return results


def monthly_mfe_series(trade_details: list[dict]) -> list[tuple]:
    """按月汇总MFE（有利波动%）均值，返回[(年月, 月度MFE均值), ...]。"""
    monthly: dict[tuple, list] = {}
    for t in trade_details:
        if t["mfe_pct"] is None:
            continue
        ym = (t["date"].year, t["date"].month)
        monthly.setdefault(ym, []).append(t["mfe_pct"])
    result = sorted((ym, statistics.mean(vals)) for ym, vals in monthly.items())
    dates = [t["date"] for t in trade_details if t["mfe_pct"] is not None]
    if dates:
        result = drop_incomplete_last_month(result, max(dates))
    return result


def rolling_mfe_ratio(monthly_mfe: list[tuple], hist_mean: float, window: int = 3) -> list[dict]:
    """滚动window个月MFE均值 / 全历史均值，返回比值序列。"""
    results = []
    for i in range(window - 1, len(monthly_mfe)):
        wm = monthly_mfe[i - window + 1: i + 1]
        vals = [v for _, v in wm]
        recent_mean = statistics.mean(vals)
        ratio = recent_mean / hist_mean if hist_mean else None
        results.append({"end_month": wm[-1][0], "mfe_ratio": ratio})
    return results


CUSUM_K_OPTIONS = [0.5]  # 允许漂移（单位：全历史标准差），标准CUSUM惯例取0.5
CUSUM_H_OPTIONS = [3.0, 4.0, 5.0]  # 报警阈值（单位：全历史标准差），扫描不同敏感度


def cusum_breach_months(monthly_pnl: list[tuple], k_sigma: float, h_sigma: float) -> list[tuple]:
    """标准单侧CUSUM变点检测：用全历史均值/标准差标准化月度净盈亏，检测均值下移。

    C_t = max(0, C_{t-1} - (x_t - mu)/sigma - k)，C_t > h 时报警，报警后C_t重置为0
    （标准做法，避免一次大幅下移后持续触发同一个变点）。
    k/h 单位均为标准差倍数，这是CUSUM的标准参数化方式，不是任意量级。
    返回触发报警的(年月)列表，口径是"检测到即报警"，不做连续N月要求（变点检测
    本身已经是对"持续性漂移"的检测，再叠加连续月数会重复惩罚，与盈亏比/胜率/
    Sharpe/MFE那种"原始值直接比较阈值"的指标性质不同）。
    """
    pnls = [p for _, p in monthly_pnl]
    if len(pnls) < 2:
        return []
    mu = statistics.mean(pnls)
    sigma = statistics.stdev(pnls)
    if sigma == 0:
        return []
    alarms = []
    c = 0.0
    for (ym, p) in monthly_pnl:
        z = (p - mu) / sigma
        c = max(0.0, c - z - k_sigma)
        if c > h_sigma:
            alarms.append(ym)
            c = 0.0  # 报警后重置，避免同一段漂移反复报警
    return alarms


def scan_cusum(per_strategy_pnl: dict[str, list[tuple]]) -> None:
    print("\n===== 指标：CUSUM变点检测（月度净盈亏序列，检测均值下移） =====")
    print("（k=允许漂移，h=报警阈值，单位均为全历史标准差倍数；报警后重置，不叠加连续月数要求）")
    print(f"{'k':>6} {'h':>6} {'独立触发事件总数':>16}  各策略明细")
    for k_sigma in CUSUM_K_OPTIONS:
        for h_sigma in CUSUM_H_OPTIONS:
            total = 0
            detail = []
            for key, monthly_pnl in per_strategy_pnl.items():
                alarms = cusum_breach_months(monthly_pnl, k_sigma, h_sigma)
                if alarms:
                    detail.append(f"{key}:{len(alarms)}")
                total += len(alarms)
            detail_str = ", ".join(detail) if detail else "无"
            print(f"{k_sigma:>6.1f} {h_sigma:>6.1f} {total:>16}  {detail_str}")


def list_cusum_events(per_strategy_pnl: dict[str, list[tuple]], k_sigma: float, h_sigma: float) -> None:
    """列出指定(k,h)组合下的具体CUSUM报警月份，供回算脚本取用。"""
    import calendar
    print(f"\n===== 明细：CUSUM k={k_sigma} h={h_sigma} 的具体报警事件 =====")
    for key, monthly_pnl in sorted(per_strategy_pnl.items()):
        alarms = cusum_breach_months(monthly_pnl, k_sigma, h_sigma)
        for (y, m) in alarms:
            last_day = calendar.monthrange(y, m)[1]
            print(f'  ("{key}", datetime.date({y}, {m}, {last_day})),  # CUSUM报警')


def count_independent_events(rows: list[dict], breach_fn, consecutive: int) -> list[tuple]:
    """按「连续N个月破线」规则数独立触发事件，返回(确认月份, 段长)列表。"""
    breach_flags = [breach_fn(r) for r in rows]
    events = []
    run_len = 0
    for i, flag in enumerate(breach_flags):
        if flag:
            run_len += 1
            if run_len == consecutive:
                events.append((rows[i]["end_month"], run_len))
        else:
            run_len = 0
    return events


def scan_pl_ratio(per_strategy_rows: dict[str, list[dict]]) -> None:
    print("===== 指标：滚动3个月盈亏比 =====")
    print(f"{'阈值':>6} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for threshold in PL_RATIO_THRESHOLDS:
        for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
            total = 0
            detail = []
            for key, rows in per_strategy_rows.items():
                events = count_independent_events(
                    rows, lambda r, t=threshold: r["pl_ratio"] < t, consecutive)
                if events:
                    detail.append(f"{key}:{len(events)}")
                total += len(events)
            detail_str = ", ".join(detail) if detail else "无"
            print(f"{threshold:>6.2f} {consecutive:>8} {total:>16}  {detail_str}")


def scan_win_rate(per_strategy_rows: dict[str, list[dict]]) -> None:
    print("\n===== 指标：滚动3个月胜率（相对该策略历史均值的比例阈值） =====")
    # 历史均值胜率按策略单独算，不用 health_monitor.md 的固定27%/22%，
    # 因为这里要测试"相对阈值"本身是否比固定阈值能筛出更干净的样本
    hist_means = {}
    for key, rows in per_strategy_rows.items():
        if rows:
            hist_means[key] = statistics.mean(r["win_rate"] for r in rows)

    print(f"{'相对阈值':>8} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for rel in WIN_RATE_REL_THRESHOLDS:
        for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
            total = 0
            detail = []
            for key, rows in per_strategy_rows.items():
                if key not in hist_means:
                    continue
                abs_threshold = hist_means[key] * rel
                events = count_independent_events(
                    rows, lambda r, t=abs_threshold: r["win_rate"] < t, consecutive)
                if events:
                    detail.append(f"{key}:{len(events)}")
                total += len(events)
            detail_str = ", ".join(detail) if detail else "无"
            print(f"{rel:>8.2f} {consecutive:>8} {total:>16}  {detail_str}")


def scan_joint(per_strategy_rows: dict[str, list[dict]]) -> None:
    """盈亏比 + 胜率联合判据：两者同时破线才算触发。"""
    print("\n===== 联合判据：盈亏比 < 阈值 且 胜率 < 历史均值*比例，同时满足才算破线 =====")
    hist_means = {}
    for key, rows in per_strategy_rows.items():
        if rows:
            hist_means[key] = statistics.mean(r["win_rate"] for r in rows)

    print(f"{'盈亏比阈值':>10} {'胜率相对阈值':>12} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for pl_t in [1.58, 1.80, 2.00]:
        for wr_rel in [0.70, 0.80, 0.90]:
            for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
                total = 0
                detail = []
                for key, rows in per_strategy_rows.items():
                    if key not in hist_means:
                        continue
                    wr_abs = hist_means[key] * wr_rel

                    def breach(r, pl_t=pl_t, wr_abs=wr_abs):
                        return r["pl_ratio"] < pl_t and r["win_rate"] < wr_abs

                    events = count_independent_events(rows, breach, consecutive)
                    if events:
                        detail.append(f"{key}:{len(events)}")
                    total += len(events)
                detail_str = ", ".join(detail) if detail else "无"
                print(f"{pl_t:>10.2f} {wr_rel:>12.2f} {consecutive:>8} {total:>16}  {detail_str}")


def scan_sharpe(per_strategy_sharpe: dict[str, list[dict]]) -> None:
    print("\n===== 指标：滚动12个月Sharpe（月度净盈亏均值/标准差，不年化） =====")
    print(f"{'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
        total = 0
        detail = []
        for key, rows in per_strategy_sharpe.items():
            events = count_independent_events(
                rows, lambda r: r["sharpe"] is not None and r["sharpe"] < 0, consecutive)
            if events:
                detail.append(f"{key}:{len(events)}")
            total += len(events)
        detail_str = ", ".join(detail) if detail else "无"
        print(f"{consecutive:>8} {total:>16}  {detail_str}")


def scan_mfe(per_strategy_mfe: dict[str, list[dict]]) -> None:
    print("\n===== 指标：滚动3个月MFE均值 / 全历史均值 =====")
    print(f"{'相对阈值':>8} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for rel in MFE_RATIO_REL_THRESHOLDS:
        for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
            total = 0
            detail = []
            for key, rows in per_strategy_mfe.items():
                events = count_independent_events(
                    rows, lambda r, t=rel: r["mfe_ratio"] is not None and r["mfe_ratio"] < t, consecutive)
                if events:
                    detail.append(f"{key}:{len(events)}")
                total += len(events)
            detail_str = ", ".join(detail) if detail else "无"
            print(f"{rel:>8.2f} {consecutive:>8} {total:>16}  {detail_str}")


def build_monthly_value_map(per_strategy_rows: dict[str, list[dict]], field: str) -> dict[str, dict]:
    """把 [{"end_month":..., field:...}, ...] 转成 {策略: {end_month: 值}}，供两两组合对齐用。"""
    return {key: {r["end_month"]: r[field] for r in rows} for key, rows in per_strategy_rows.items()}


def scan_joint_2metric(name_a: str, map_a: dict[str, dict], breach_a,
                        name_b: str, map_b: dict[str, dict], breach_b,
                        thresholds_a: list, thresholds_b: list) -> None:
    """通用两指标联合判据扫描：两者按end_month对齐后同时满足才算破线。

    breach_a/breach_b 签名为 (value, threshold) -> bool，用于统一不同指标的比较方向。
    """
    print(f"\n===== 联合判据：{name_a} 阈值扫描 x {name_b} 阈值扫描，同时满足才算破线 =====")
    print(f"{name_a+'阈值':>12} {name_b+'阈值':>12} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for t_a in thresholds_a:
        for t_b in thresholds_b:
            for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
                total = 0
                detail = []
                for key in map_a:
                    if key not in map_b:
                        continue
                    common_months = sorted(set(map_a[key]) & set(map_b[key]))
                    if not common_months:
                        continue
                    synth_rows = [{"end_month": m} for m in common_months]

                    def breach(r, key=key, t_a=t_a, t_b=t_b):
                        m = r["end_month"]
                        va = map_a[key][m]
                        vb = map_b[key][m]
                        return breach_a(va, t_a) and breach_b(vb, t_b)

                    events = count_independent_events(synth_rows, breach, consecutive)
                    if events:
                        detail.append(f"{key}:{len(events)}")
                    total += len(events)
                detail_str = ", ".join(detail) if detail else "无"
                print(f"{t_a:>12.2f} {t_b:>12.2f} {consecutive:>8} {total:>16}  {detail_str}")


def list_joint_2metric_events(name_a: str, map_a: dict[str, dict], breach_a, t_a,
                               name_b: str, map_b: dict[str, dict], breach_b, t_b,
                               consecutive: int) -> None:
    """列出指定两指标联合判据阈值组合下的具体触发事件，供回算脚本取用。"""
    import calendar
    print(f"\n===== 明细：{name_a}阈值={t_a}，{name_b}阈值={t_b}，连续{consecutive}个月 的具体触发事件 =====")
    for key in sorted(map_a):
        if key not in map_b:
            continue
        common_months = sorted(set(map_a[key]) & set(map_b[key]))
        if not common_months:
            continue
        synth_rows = [{"end_month": m} for m in common_months]

        def breach(r, key=key):
            m = r["end_month"]
            return breach_a(map_a[key][m], t_a) and breach_b(map_b[key][m], t_b)

        events = count_independent_events(synth_rows, breach, consecutive)
        for end_month, run_len in events:
            y, m = end_month
            last_day = calendar.monthrange(y, m)[1]
            print(f'    ("{key}", datetime.date({y}, {m}, {last_day})),  # 段长{run_len}月')


def scan_joint_3metric(per_strategy_rows: dict[str, list[dict]],
                        per_strategy_sharpe: dict[str, list[dict]],
                        per_strategy_mfe: dict[str, list[dict]]) -> None:
    """盈亏比 + Sharpe + MFE比值三指标联合判据：按月末对齐后要求同时满足。

    三个指标滚动窗口不同（盈亏比/MFE是3个月，Sharpe是12个月），用end_month对齐，
    Sharpe序列更短（需要更长历史才有第一个数据点），对齐后能联合判断的月份数更少。
    """
    print("\n===== 联合判据：盈亏比<阈值 且 Sharpe<0 且 MFE比值<阈值，三者同时满足才算破线 =====")
    print("（按end_month对齐三个序列，Sharpe需12个月历史才有首个数据点，可联合判断的月份本就少于单指标）")

    for key in per_strategy_rows:
        if key not in per_strategy_sharpe or key not in per_strategy_mfe:
            continue
        pl_by_month = {r["end_month"]: r["pl_ratio"] for r in per_strategy_rows[key]}
        sharpe_by_month = {r["end_month"]: r["sharpe"] for r in per_strategy_sharpe[key]}
        mfe_by_month = {r["end_month"]: r["mfe_ratio"] for r in per_strategy_mfe[key]}
        common_months = sorted(set(pl_by_month) & set(sharpe_by_month) & set(mfe_by_month))
        if common_months:
            print(f"  {key}：三指标均有数据的月份数 = {len(common_months)}（Sharpe序列限制了对齐窗口）")

    print(f"\n{'盈亏比阈值':>10} {'MFE相对阈值':>12} {'连续月数':>8} {'独立触发事件总数':>16}  各策略明细")
    for pl_t in [1.80, 2.00]:
        for mfe_rel in [0.60, 0.70]:
            for consecutive in CONSECUTIVE_MONTHS_OPTIONS:
                total = 0
                detail = []
                for key in per_strategy_rows:
                    if key not in per_strategy_sharpe or key not in per_strategy_mfe:
                        continue
                    pl_by_month = {r["end_month"]: r["pl_ratio"] for r in per_strategy_rows[key]}
                    sharpe_by_month = {r["end_month"]: r["sharpe"] for r in per_strategy_sharpe[key]}
                    mfe_by_month = {r["end_month"]: r["mfe_ratio"] for r in per_strategy_mfe[key]}
                    common_months = sorted(set(pl_by_month) & set(sharpe_by_month) & set(mfe_by_month))
                    if not common_months:
                        continue
                    synth_rows = [{"end_month": m} for m in common_months]

                    def breach(r, pl_t=pl_t, mfe_rel=mfe_rel,
                               pl_by_month=pl_by_month, sharpe_by_month=sharpe_by_month, mfe_by_month=mfe_by_month):
                        m = r["end_month"]
                        sh = sharpe_by_month[m]
                        mf = mfe_by_month[m]
                        return (pl_by_month[m] < pl_t and sh is not None and sh < 0
                                and mf is not None and mf < mfe_rel)

                    events = count_independent_events(synth_rows, breach, consecutive)
                    if events:
                        detail.append(f"{key}:{len(events)}")
                    total += len(events)
                detail_str = ", ".join(detail) if detail else "无"
                print(f"{pl_t:>10.2f} {mfe_rel:>12.2f} {consecutive:>8} {total:>16}  {detail_str}")


def list_events_detail(per_strategy_rows: dict[str, list[dict]], threshold: float, consecutive: int) -> None:
    """列出指定(阈值,连续月数)组合下的具体触发事件，供回算脚本取用。"""
    import calendar
    print(f"\n===== 明细：盈亏比阈值={threshold}，连续{consecutive}个月 的具体触发事件 =====")
    for key, rows in sorted(per_strategy_rows.items()):
        events = count_independent_events(
            rows, lambda r, t=threshold: r["pl_ratio"] < t, consecutive)
        for end_month, run_len in events:
            y, m = end_month
            last_day = calendar.monthrange(y, m)[1]
            print(f'  ("{key}", datetime.date({y}, {m}, {last_day})),  # 确认月末，段长{run_len}个月')


def main():
    parser = argparse.ArgumentParser(description="触发条件网格扫描")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"))
    parser.add_argument("--metric",
                        choices=["pl_ratio", "win_rate", "joint", "sharpe", "mfe", "cusum",
                                 "joint_pl_sharpe", "joint_pl_mfe", "joint_sharpe_mfe", "joint3", "all"],
                        default="all")
    parser.add_argument("--list-events", action="store_true", help="列出指定阈值+连续月数的具体触发事件")
    parser.add_argument("--threshold", type=float, default=2.00)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--list-cusum", action="store_true", help="列出指定k/h组合的具体CUSUM报警事件")
    parser.add_argument("--cusum-k", type=float, default=0.5)
    parser.add_argument("--cusum-h", type=float, default=4.0)
    parser.add_argument("--list-joint2", choices=["pl_sharpe", "pl_mfe", "sharpe_mfe"],
                        help="列出指定两指标联合判据下的具体触发事件（需配合--threshold-a/--threshold-b）")
    parser.add_argument("--threshold-a", type=float, default=2.00)
    parser.add_argument("--threshold-b", type=float, default=0.70)
    args = parser.parse_args()

    files = scan_files_auto(args.dir)
    deployed = {k: fp for k, fp in files.items() if k.endswith(DEPLOYED_SUFFIXES)}
    print(f"实盘部署范围（v2+v3_205m）共 {len(deployed)} 个策略：{sorted(deployed.keys())}\n")

    per_strategy_rows = {}
    per_strategy_sharpe = {}
    per_strategy_mfe = {}
    per_strategy_pnl = {}
    for key, fp in deployed.items():
        trades = read_trade_pnls(fp)
        monthly = monthly_win_loss(trades)
        if trades:
            # 剔除不完整的最新月份，避免月中运行时把"数据还没走完的月份"误判为真实触发月
            last_trade_date = max(d for d, _ in trades)
            last_ym = max(monthly.keys())
            complete = drop_incomplete_last_month([(last_ym, None)], last_trade_date)
            if not complete:
                monthly = {m: v for m, v in monthly.items() if m != last_ym}
        per_strategy_rows[key] = rolling_pl_ratio(monthly, window=3)

        monthly_pnl = monthly_pnl_series(trades)
        per_strategy_pnl[key] = monthly_pnl
        per_strategy_sharpe[key] = rolling_sharpe(monthly_pnl, window=SHARPE_WINDOW_MONTHS)

        detail = read_trade_detail(fp)
        mfe_vals = [t["mfe_pct"] for t in detail if t["mfe_pct"] is not None]
        if mfe_vals:
            hist_mean = statistics.mean(mfe_vals)
            monthly_mfe = monthly_mfe_series(detail)
            per_strategy_mfe[key] = rolling_mfe_ratio(monthly_mfe, hist_mean, window=3)

    pl_map = build_monthly_value_map(per_strategy_rows, "pl_ratio")
    sharpe_map = build_monthly_value_map(per_strategy_sharpe, "sharpe")
    mfe_map = build_monthly_value_map(per_strategy_mfe, "mfe_ratio")
    lt = lambda v, t: v is not None and v < t  # noqa: E731，三个两组合共用的"小于阈值才算破线"判定

    if args.list_events:
        list_events_detail(per_strategy_rows, args.threshold, args.consecutive)
        return
    if args.list_cusum:
        list_cusum_events(per_strategy_pnl, args.cusum_k, args.cusum_h)
        return
    if args.list_joint2:
        pairs = {
            "pl_sharpe": ("盈亏比", pl_map, "Sharpe", sharpe_map),
            "pl_mfe": ("盈亏比", pl_map, "MFE比值", mfe_map),
            "sharpe_mfe": ("Sharpe", sharpe_map, "MFE比值", mfe_map),
        }
        name_a, map_a, name_b, map_b = pairs[args.list_joint2]
        list_joint_2metric_events(name_a, map_a, lt, args.threshold_a,
                                   name_b, map_b, lt, args.threshold_b, args.consecutive)
        return

    if args.metric in ("pl_ratio", "all"):
        scan_pl_ratio(per_strategy_rows)
    if args.metric in ("win_rate", "all"):
        scan_win_rate(per_strategy_rows)
    if args.metric in ("joint", "all"):
        scan_joint(per_strategy_rows)
    if args.metric in ("sharpe", "all"):
        scan_sharpe(per_strategy_sharpe)
    if args.metric in ("mfe", "all"):
        scan_mfe(per_strategy_mfe)
    if args.metric in ("cusum", "all"):
        scan_cusum(per_strategy_pnl)
    if args.metric in ("joint_pl_sharpe", "all"):
        scan_joint_2metric("盈亏比", pl_map, lt, "Sharpe", sharpe_map, lt,
                           [1.58, 1.80, 2.00], [0.0])
    if args.metric in ("joint_pl_mfe", "all"):
        scan_joint_2metric("盈亏比", pl_map, lt, "MFE比值", mfe_map, lt,
                           [1.58, 1.80, 2.00], [0.60, 0.70, 0.80])
    if args.metric in ("joint_sharpe_mfe", "all"):
        scan_joint_2metric("Sharpe", sharpe_map, lt, "MFE比值", mfe_map, lt,
                           [0.0], [0.60, 0.70, 0.80])
    if args.metric in ("joint3", "all"):
        scan_joint_3metric(per_strategy_rows, per_strategy_sharpe, per_strategy_mfe)


if __name__ == "__main__":
    main()
