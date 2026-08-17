"""
中证500指数样本股定期调整效应诊断（事件研究，仅测核心前提）

背景：项目此前六轮40+方向+第七/八轮候选全部是"因子选股"类方案（固定权重
因子组合），核心瓶颈是这类方案在A股风格切换下必然衰减。指数样本股调整效应
是完全不同的信号逻辑——不是选股因子，是被动基金跟踪指数导致的资金流事件：
中证指数公司公告调入调出名单后，跟踪该指数的被动基金需要在生效日前买入
调入股/卖出调出股，这个提前建仓的资金流可能推高调入股价格（"指数效应"，
海外市场文献已有大量记录，如标普500的index effect）。

数据与方法：
- 调入/调出名单：`hs500_members.parquet` 月末快照diff（用hs500而非hs300，
  因为中证500样本调整数量更稳定，历次固定约50只调入50只调出）。
  该文件由 fetch_index_members.py 生成，需要先跑一遍中证500版本
  （INDEX_CODE=399905.SZ）才有数据，若不存在则本脚本报错退出。
- 生效日：中证指数公司规则是"6月/12月第二个星期五的下一交易日"生效，
  用 trade_cal 计算历年该日期（已实测验证2016-2025年20次调整规律稳定，
  不依赖index_weight快照本身推断，是公开规则的直接计算）。
- 公告日近似：中证指数公司规则是生效日前两周左右公告名单，本脚本用
  "生效日前10个交易日"作为窗口起点近似代表公告日附近（不是精确公告日，
  只是近似窗口，如果信号在这个窗口内不显著，可判定"提前买入"效应不存在，
  不需要更精确的公告日就能得出结论）。
- 基准：中证500指数自身点位（index_daily），不用无效应对照组（YAGNI，
  指数本身就是最自然的基准）。

先测前提（.claude/lessons.md 第99条方法论）：只算"公告日附近到生效日"窗口
内调入组/调出组相对指数的超额收益，这是"指数效应"能否存在的最核心信号。
如果这个信号不显著，直接判定证伪，不再做完整的组合构建/回测。

用法：
  cd a_stock/backtest
  python event_index_rebalance.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import init_pro, DATA_DIR, STOCK_DIR

HS500_MEMBERS_FILE = DATA_DIR / "hs500_members.parquet"
INDEX_CODE = "000905.SH"  # 中证500指数点位（区别于399905.SZ的成分股代码空间）
ANNOUNCE_LAG_TRADING_DAYS = 10  # 生效日前N个交易日近似代表公告日附近

START_YEAR = 2016
END_YEAR = 2025


def get_effective_date(trade_days: pd.DatetimeIndex, year: int, month: int) -> pd.Timestamp:
    """中证指数公司规则：当月第二个周五的下一交易日生效"""
    fridays = [d for d in pd.date_range(f"{year}-{month:02d}-01", f"{year}-{month:02d}-28") if d.weekday() == 4]
    second_friday = fridays[1]
    nxt = trade_days[trade_days > second_friday]
    if len(nxt) == 0:
        return None
    return nxt[0]


def load_adjustment_events(members: pd.DataFrame) -> list[dict]:
    """按6月/12月月末快照diff识别历次调入调出名单"""
    snap_dates = sorted(members["trade_date"].unique())
    events = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in [6, 12]:
            target = [d for d in snap_dates if pd.Timestamp(d).year == year and pd.Timestamp(d).month == month]
            prev_candidates = [d for d in snap_dates if pd.Timestamp(d) < min(target, default=pd.Timestamp.max)]
            if not target or not prev_candidates:
                continue
            cur_date = target[0]
            prev_date = max(prev_candidates)
            cur_set = set(members[members["trade_date"] == cur_date]["con_code"])
            prev_set = set(members[members["trade_date"] == prev_date]["con_code"])
            added = cur_set - prev_set
            removed = prev_set - cur_set
            if not added and not removed:
                continue
            events.append({
                "year": year, "month": month,
                "snap_date": pd.Timestamp(cur_date),
                "added": sorted(added), "removed": sorted(removed),
            })
    return events


def load_index_daily(pro) -> pd.Series:
    df = pro.index_daily(ts_code=INDEX_CODE, start_date="20160101", end_date="20261231",
                          fields="trade_date,close")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].sort_index()


def load_stock_close(ts_code: str) -> pd.Series:
    path = STOCK_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path, columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].sort_index()


def group_window_return(codes: list[str], window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """等权组合从window_start到window_end的累计收益（用各股在窗口内实际可交易的首末价）"""
    rets = []
    for code in codes:
        s = load_stock_close(code)
        s = s.loc[window_start:window_end]
        if len(s) < 2:
            continue
        rets.append(s.iloc[-1] / s.iloc[0] - 1)
    if not rets:
        return np.nan
    return float(np.mean(rets))


def index_window_return(index_close: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    s = index_close.loc[window_start:window_end]
    if len(s) < 2:
        return np.nan
    return s.iloc[-1] / s.iloc[0] - 1


# ── 净收益核算（T+1建仓 + 不同退出时点 + 扣成本） ──────────────────────
#
# 成本口径：采用 .claude/rules/trading-standards.md 的项目级统一标准——
# 单次完整回合成本约0.164%（佣金万1双边+印花税千1卖出+过户费万0.2双边+
# 滑点万2双边）。factor_multi_backtest_v2.py 里 COST_PER_TRADE+STAMP_DUTY
# 合计约0.4%是该脚本的历史遗留参数，不采用——trading-standards.md 是本项目
# 交易成本的唯一权威来源，各回测脚本应向其收敛而非各自为政。
ROUND_TRIP_COST = 0.00164

# 退出时点相对生效日的交易日偏移：0=生效日当天，-1/-2=提前1/2个交易日退出
# （测试"被动基金集中调仓后价格回吐"是否在生效日前就已发生，提前退出能否
# 避开回吐段）
EXIT_OFFSETS = [0, -1, -2]


def shift_trading_day(trade_days: pd.DatetimeIndex, date: pd.Timestamp, offset: int) -> pd.Timestamp:
    """把日期沿交易日历移动offset个交易日（offset可为负）"""
    pos = trade_days.get_indexer([date])[0]
    if pos == -1:
        pos = trade_days.searchsorted(date)
    new_pos = pos + offset
    if new_pos < 0 or new_pos >= len(trade_days):
        return None
    return trade_days[new_pos]


def entry_price(ts_code: str, entry_date: pd.Timestamp) -> float:
    """T+1建仓价：entry_date当天及之后最近一个交易日的收盘价"""
    s = load_stock_close(ts_code)
    s = s.loc[entry_date:]
    if s.empty:
        return np.nan
    return s.iloc[0]


def exit_price(ts_code: str, exit_date: pd.Timestamp) -> float:
    """退出价：exit_date当天及之前最近一个交易日的收盘价"""
    s = load_stock_close(ts_code)
    s = s.loc[:exit_date]
    if s.empty:
        return np.nan
    return s.iloc[-1]


def group_net_return(codes: list[str], entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> tuple[float, int]:
    """等权组合从entry_date到exit_date的累计收益（gross，未扣成本），返回(收益, 有效个股数)"""
    rets = []
    for code in codes:
        p_in = entry_price(code, entry_date)
        p_out = exit_price(code, exit_date)
        if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
            continue
        rets.append(p_out / p_in - 1)
    if not rets:
        return np.nan, 0
    return float(np.mean(rets)), len(rets)


def net_return_analysis(events: list[dict], trade_days: pd.DatetimeIndex, index_close: pd.Series) -> pd.DataFrame:
    """
    对每次调整事件、每个退出时点偏移，计算T+1建仓的净超额收益（扣完整回合成本）。
    只做调入组（做多逻辑）——调出组是"规避买入"而非"做空"，A股个股不能做空
    （a_stock/CLAUDE.md），调出组没有可交易的净收益核算对象，此函数不涉及。
    """
    rows = []
    for ev in events:
        eff_date = get_effective_date(trade_days, ev["year"], ev["month"])
        if eff_date is None:
            continue
        pos = trade_days.get_indexer([eff_date])[0]
        if pos < ANNOUNCE_LAG_TRADING_DAYS:
            continue
        window_start = trade_days[pos - ANNOUNCE_LAG_TRADING_DAYS]
        entry_date = shift_trading_day(trade_days, window_start, 1)  # T+1建仓
        if entry_date is None:
            continue

        for offset in EXIT_OFFSETS:
            exit_date = shift_trading_day(trade_days, eff_date, offset)
            if exit_date is None or exit_date <= entry_date:
                continue
            gross, n_valid = group_net_return(ev["added"], entry_date, exit_date)
            idx_ret = index_window_return(index_close, entry_date, exit_date)
            if pd.isna(gross) or pd.isna(idx_ret):
                continue
            gross_excess = gross - idx_ret
            net_excess = gross_excess - ROUND_TRIP_COST
            rows.append({
                "year": ev["year"], "month": ev["month"], "exit_offset": offset,
                "entry_date": entry_date.date(), "exit_date": exit_date.date(),
                "n_valid": n_valid, "n_added": len(ev["added"]),
                "gross_excess": gross_excess, "net_excess": net_excess,
            })
    return pd.DataFrame(rows)


# ── 涨跌停可执行性检查 ──────────────────────────────────────────
#
# stock_daily 本地数据只有 open/high/low/close/vol/amount，没有专门的涨跌停
# 价字段（已实测确认，见600688.SH.parquet样例），用涨幅近似判断：
# 创业板/科创板（300/301/688开头）涨跌停幅度20%，用9.5%~19.5%区间容易误判，
# 阈值取19.5%；其余主板/中小板阈值取9.5%（留0.5%缓冲吸收复权价格误差）。
def is_star_or_chinext(ts_code: str) -> bool:
    code = ts_code.split(".")[0]
    return code.startswith(("300", "301", "688"))


def limit_up_blocked(ts_code: str, entry_date: pd.Timestamp) -> bool:
    """T+1建仓当天是否涨停（无法以当日价格买入）：用当日涨幅是否达到板块涨停幅度近似判断"""
    path = STOCK_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return False
    df = pd.read_parquet(path, columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")["close"].sort_index()
    df = df.loc[:entry_date]
    if len(df) < 2:
        return False
    pct = df.iloc[-1] / df.iloc[-2] - 1
    threshold = 0.195 if is_star_or_chinext(ts_code) else 0.095
    return pct >= threshold


def limit_up_check(events: list[dict], trade_days: pd.DatetimeIndex) -> pd.DataFrame:
    """对调入组逐次事件检查T+1建仓日涨停不可执行的个股比例"""
    rows = []
    for ev in events:
        eff_date = get_effective_date(trade_days, ev["year"], ev["month"])
        if eff_date is None:
            continue
        pos = trade_days.get_indexer([eff_date])[0]
        if pos < ANNOUNCE_LAG_TRADING_DAYS:
            continue
        window_start = trade_days[pos - ANNOUNCE_LAG_TRADING_DAYS]
        entry_date = shift_trading_day(trade_days, window_start, 1)
        if entry_date is None:
            continue
        blocked = [c for c in ev["added"] if limit_up_blocked(c, entry_date)]
        rows.append({
            "year": ev["year"], "month": ev["month"], "entry_date": entry_date.date(),
            "n_added": len(ev["added"]), "n_blocked": len(blocked),
            "blocked_pct": len(blocked) / len(ev["added"]) if ev["added"] else np.nan,
        })
    return pd.DataFrame(rows)


# ── 容量与年化贡献量级估算 ──────────────────────────────────────
#
# 这是低频事件策略（每年仅2次调整，每次持有约9-10个交易日），不是全时段
# 持仓策略。年化贡献估算口径：单次事件净超额收益 × 每年事件数(2)，
# 代表"若每次事件都用同一笔资金参与"这部分资金的年化超额贡献
# （不代表整个组合的年化收益，因为这笔资金一年里大部分时间空闲，
# 可与其他策略共用，YAGNI——不在本脚本里设计资金复用方案）。
EVENTS_PER_YEAR = 2


def load_daily_amount(ts_code: str) -> pd.Series:
    path = STOCK_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path, columns=["trade_date", "amount"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["amount"].sort_index()


def capacity_estimate(events: list[dict], trade_days: pd.DatetimeIndex) -> float:
    """
    估算容量：调入组个股在建仓窗口内的平均日成交额（tushare amount单位为千元），
    按"不超过单日成交额10%避免明显冲击成本"这一行业惯用经验阈值折算可承载资金量。
    """
    daily_amounts = []
    for ev in events:
        eff_date = get_effective_date(trade_days, ev["year"], ev["month"])
        if eff_date is None:
            continue
        pos = trade_days.get_indexer([eff_date])[0]
        if pos < ANNOUNCE_LAG_TRADING_DAYS:
            continue
        window_start = trade_days[pos - ANNOUNCE_LAG_TRADING_DAYS]
        entry_date = shift_trading_day(trade_days, window_start, 1)
        if entry_date is None:
            continue
        for code in ev["added"]:
            s = load_daily_amount(code)
            s = s.loc[entry_date:eff_date]
            if not s.empty:
                daily_amounts.append(s.mean())
    if not daily_amounts:
        return np.nan
    avg_daily_amount_thousand_yuan = float(np.mean(daily_amounts))
    avg_daily_amount_yuan = avg_daily_amount_thousand_yuan * 1000
    # 50只等权分摊，单只不超过其自身日均成交额10%
    per_stock_capacity = avg_daily_amount_yuan * 0.10
    return per_stock_capacity * 50  # 50只同时建仓的组合总容量


def summarize_capacity(net_df: pd.DataFrame, limit_df: pd.DataFrame, capacity_yuan: float) -> None:
    print("\n=== 容量与年化贡献量级估算 ===")
    best = net_df[net_df["exit_offset"] == -1]
    gross_mean = best["gross_excess"].mean()
    net_mean = best["net_excess"].mean()
    blocked_pct = limit_df["n_blocked"].sum() / limit_df["n_added"].sum()
    net_after_limit = net_mean * (1 - blocked_pct)

    print(f"单次事件（退出偏移-1，即生效日前1个交易日）：")
    print(f"  gross超额收益均值 = {gross_mean:+.4%}")
    print(f"  net超额收益均值（扣完整回合成本{ROUND_TRIP_COST:.3%}） = {net_mean:+.4%}")
    print(f"  net超额收益均值（再扣涨停不可执行比例{blocked_pct:.1%}折算） = {net_after_limit:+.4%}")
    print(f"\n年化贡献估算（{EVENTS_PER_YEAR}次事件/年，同一笔资金参与）：")
    print(f"  gross年化贡献 ≈ {gross_mean * EVENTS_PER_YEAR:+.4%}")
    print(f"  net年化贡献 ≈ {net_mean * EVENTS_PER_YEAR:+.4%}")
    print(f"  net年化贡献（含涨跌停折算） ≈ {net_after_limit * EVENTS_PER_YEAR:+.4%}")
    if pd.notna(capacity_yuan):
        print(f"\n容量估算（单只个股不超日均成交额10%，50只等权）：约 {capacity_yuan / 1e8:.2f} 亿元/次事件")
    else:
        print("\n容量估算：无有效成交额数据")


def summarize_limit_up(df: pd.DataFrame) -> None:
    print("\n=== 涨跌停可执行性检查（调入组T+1建仓日）===")
    total_added = df["n_added"].sum()
    total_blocked = df["n_blocked"].sum()
    print(f"总调入个股数={total_added}  涨停不可执行数={total_blocked}  "
          f"整体不可执行比例={total_blocked / total_added:.1%}")
    print(f"逐事件不可执行比例：均值={df['blocked_pct'].mean():.1%}  "
          f"最大={df['blocked_pct'].max():.1%}  "
          f"发生涨停的事件数={ (df['n_blocked'] > 0).sum() }/{len(df)}")


def summarize_net_return(df: pd.DataFrame) -> None:
    print("\n=== 净收益核算（T+1建仓，扣完整回合成本%.3f%%）===" % (ROUND_TRIP_COST * 100))
    for offset in EXIT_OFFSETS:
        sub = df[df["exit_offset"] == offset]["net_excess"].dropna()
        if sub.empty:
            print(f"\n退出偏移{offset:+d}个交易日：无有效数据")
            continue
        mean = sub.mean()
        same_sign = (np.sign(sub) == np.sign(mean)).mean() if mean != 0 else 0.0
        t_stat, p_val = stats.ttest_1samp(sub, 0)
        label = "生效日当天" if offset == 0 else f"生效日前{-offset}个交易日"
        print(f"退出={label}（offset={offset:+d}）：净超额收益均值={mean:+.4%}  "
              f"同向占比={same_sign:.1%}  n={len(sub)}事件  t={t_stat:.2f}  p={p_val:.3f}")


def main():
    if not HS500_MEMBERS_FILE.exists():
        print(f"缺少 {HS500_MEMBERS_FILE}，无法识别历次调入调出名单，脚本退出")
        return

    pro = init_pro()
    members = pd.read_parquet(HS500_MEMBERS_FILE)
    members["trade_date"] = pd.to_datetime(members["trade_date"])

    events = load_adjustment_events(members)
    print(f"识别到 {len(events)} 次调整事件（{START_YEAR}-{END_YEAR}）")

    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20160101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))

    index_close = load_index_daily(pro)

    rows = []
    for ev in events:
        eff_date = get_effective_date(trade_days, ev["year"], ev["month"])
        if eff_date is None:
            continue
        pos = trade_days.get_indexer([eff_date])[0]
        if pos < ANNOUNCE_LAG_TRADING_DAYS:
            continue
        window_start = trade_days[pos - ANNOUNCE_LAG_TRADING_DAYS]
        window_end = eff_date

        added_ret = group_window_return(ev["added"], window_start, window_end)
        removed_ret = group_window_return(ev["removed"], window_start, window_end)
        idx_ret = index_window_return(index_close, window_start, window_end)

        rows.append({
            "year": ev["year"], "month": ev["month"],
            "n_added": len(ev["added"]), "n_removed": len(ev["removed"]),
            "window_start": window_start.date(), "eff_date": eff_date.date(),
            "added_ret": added_ret, "removed_ret": removed_ret, "index_ret": idx_ret,
            "added_excess": added_ret - idx_ret if pd.notna(added_ret) and pd.notna(idx_ret) else np.nan,
            "removed_excess": removed_ret - idx_ret if pd.notna(removed_ret) and pd.notna(idx_ret) else np.nan,
        })

    df = pd.DataFrame(rows)
    print(f"\n有效事件数：{len(df)}")
    print(df[["year", "month", "n_added", "n_removed", "window_start", "eff_date",
              "added_excess", "removed_excess"]].to_string(index=False))

    for col, label in [("added_excess", "调入组"), ("removed_excess", "调出组")]:
        clean = df[col].dropna()
        if clean.empty:
            print(f"\n{label}：无有效数据")
            continue
        mean = clean.mean()
        same_sign = (np.sign(clean) == np.sign(mean)).mean() if mean != 0 else 0.0
        t_stat, p_val = stats.ttest_1samp(clean, 0)
        print(f"\n{label}（公告日附近至生效日窗口超额收益）：")
        print(f"  均值={mean:+.4%}  同向占比={same_sign:.1%}  n={len(clean)}事件  "
              f"t统计量={t_stat:.2f}  p值={p_val:.3f}")

    out_dir = pathlib.Path(__file__).parent / "results" / "event_index_rebalance"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "events_summary.csv", index=False)
    print(f"\n结果已保存：{out_dir / 'events_summary.csv'}")

    net_df = net_return_analysis(events, trade_days, index_close)
    summarize_net_return(net_df)
    net_df.to_csv(out_dir / "net_return_summary.csv", index=False)
    print(f"\n净收益核算结果已保存：{out_dir / 'net_return_summary.csv'}")

    limit_df = limit_up_check(events, trade_days)
    summarize_limit_up(limit_df)
    limit_df.to_csv(out_dir / "limit_up_check.csv", index=False)
    print(f"\n涨跌停检查结果已保存：{out_dir / 'limit_up_check.csv'}")

    capacity_yuan = capacity_estimate(events, trade_days)
    summarize_capacity(net_df, limit_df, capacity_yuan)


if __name__ == "__main__":
    main()
