"""
高管/大股东增持公告事件研究（事件驱动信号，仅测核心前提）

背景：指数增强新候选方向调研（a_stock/docs/research.md「指数增强策略」）
第一个待验证方向。逻辑与指数样本股调整效应（event_index_rebalance.py）
同属"事件驱动资金流"类信号，但触发方不同：这里是公司内部人（高管/大股东）
用自有资金增持，市场普遍认为是内部信息优势的信号（海外文献中的 insider
trading信号，中信证券等机构报告亦有类似研究）。

数据与方法：
- 事件识别：holder_trade.parquet（fetch_holder_trade.py 拉取）里
  in_de='IN' 的记录，同ts_code在 MERGE_WINDOW_TRADING_DAYS 个交易日内
  的多条公告合并为一次事件（合并窗口内取首条公告日期为事件日，避免
  同一波增持被计为多次独立事件重复计入统计显著性检验——已实测确认
  32.8%同ts_code+ann_date有多条记录、60.9%相邻公告间隔<=5个交易日，
  不去重会严重高估样本独立性）。
- 股票池：沪深300+中证500联合池（hs300_members.parquet+hs500_members.parquet
  历史成分股并集，1578只），复用现有数据基础设施，零额外数据成本。
- holder_type：不过滤，G(高管)/C(大股东)/P(其他个人)全部类型合并测试
  （先测全量信号是否存在，YAGNI——如果显著再考虑细分类型）。
- T+1建仓：公告日（若非交易日先顺延至下一交易日）之后再顺延一个交易日买入，
  代表"公告发布后市场能够反应的最早可交易时点"。
- 测试窗口：20/60个交易日累计超额收益（不直接对齐中信证券报告的100日
  窗口，先测短窗口看衰减速度，符合.claude/lessons.md第99条"先测前提"
  方法论——如果20日窗口就不显著，不需要再测更长窗口去凑显著性）。
- 基准：中证800（000906.SH），覆盖沪深300+中证500全部成分股，避免"每只
  股票匹配所属指数"的复杂逻辑。

复用 event_index_rebalance.py 的通用工具函数（T+1建仓价/退出价、涨跌停
可执行性检查、成交额加载、成本口径），避免重复实现同一套方法论（DRY）。

用法：
  cd a_stock/backtest
  python event_holder_increase.py
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
from event_index_rebalance import (
    shift_trading_day, entry_price, exit_price,
    is_star_or_chinext, limit_up_blocked, load_daily_amount,
    load_index_daily, index_window_return, ROUND_TRIP_COST,
)

HOLDER_TRADE_FILE = DATA_DIR / "holder_trade.parquet"
MEMBERS_FILES = [DATA_DIR / "hs300_members.parquet", DATA_DIR / "hs500_members.parquet"]
BENCHMARK_CODE = "000906.SH"  # 中证800，覆盖沪深300+中证500全部成分股

MERGE_WINDOW_TRADING_DAYS = 5  # 同股票公告合并为一次事件的交易日窗口
HOLD_WINDOWS = [20, 60]         # 测试的持有期（交易日）


def load_pool_codes() -> set[str]:
    codes = set()
    for f in MEMBERS_FILES:
        if not f.exists():
            raise FileNotFoundError(f"缺少 {f}，无法确定股票池")
        codes |= set(pd.read_parquet(f, columns=["con_code"])["con_code"].unique())
    return codes


def load_increase_events_raw() -> pd.DataFrame:
    if not HOLDER_TRADE_FILE.exists():
        raise FileNotFoundError(f"缺少 {HOLDER_TRADE_FILE}，请先运行 fetch_holder_trade.py")
    df = pd.read_parquet(HOLDER_TRADE_FILE)
    df = df[df["in_de"] == "IN"].copy()
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    return df


def dedup_to_events(raw: pd.DataFrame, pool_codes: set[str],
                     trade_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    同ts_code在MERGE_WINDOW_TRADING_DAYS个交易日内的多条公告合并为一次事件。
    合并窗口用交易日位置差（不是自然日差）判断，取每组首条公告日作为事件日。
    """
    df = raw[raw["ts_code"].isin(pool_codes)].copy()
    if df.empty:
        return pd.DataFrame(columns=["ts_code", "ann_date"])
    df = df.sort_values(["ts_code", "ann_date"])

    events = []
    for ts_code, group in df.groupby("ts_code"):
        dates = sorted(group["ann_date"].unique())
        positions = trade_days.searchsorted(dates)
        cur_start = dates[0]
        cur_last_pos = positions[0]
        for d, pos in zip(dates[1:], positions[1:]):
            if pos - cur_last_pos <= MERGE_WINDOW_TRADING_DAYS:
                cur_last_pos = pos
                continue
            events.append({"ts_code": ts_code, "ann_date": pd.Timestamp(cur_start)})
            cur_start = d
            cur_last_pos = pos
        events.append({"ts_code": ts_code, "ann_date": pd.Timestamp(cur_start)})

    return pd.DataFrame(events)


def compute_entry_date(trade_days: pd.DatetimeIndex, ann_date: pd.Timestamp) -> pd.Timestamp:
    """公告日顺延至下一交易日，再顺延一个交易日作为T+1建仓日"""
    pos = trade_days.searchsorted(ann_date)
    if pos >= len(trade_days):
        return None
    ann_trading_date = trade_days[pos]
    return shift_trading_day(trade_days, ann_trading_date, 1)


def net_return_analysis(events: pd.DataFrame, trade_days: pd.DatetimeIndex,
                         index_close: pd.Series) -> pd.DataFrame:
    """对每次事件、每个持有窗口，计算T+1建仓的净超额收益（扣完整回合成本）"""
    rows = []
    for _, ev in events.iterrows():
        entry_date = compute_entry_date(trade_days, ev["ann_date"])
        if entry_date is None:
            continue
        p_in = entry_price(ev["ts_code"], entry_date)
        if pd.isna(p_in) or p_in <= 0:
            continue

        for window in HOLD_WINDOWS:
            exit_date = shift_trading_day(trade_days, entry_date, window)
            if exit_date is None:
                continue
            p_out = exit_price(ev["ts_code"], exit_date)
            if pd.isna(p_out):
                continue
            stock_ret = p_out / p_in - 1
            idx_ret = index_window_return(index_close, entry_date, exit_date)
            if pd.isna(idx_ret):
                continue
            gross_excess = stock_ret - idx_ret
            net_excess = gross_excess - ROUND_TRIP_COST
            rows.append({
                "ts_code": ev["ts_code"], "ann_date": ev["ann_date"].date(),
                "entry_date": entry_date.date(), "exit_date": exit_date.date(),
                "window": window,
                "gross_excess": gross_excess, "net_excess": net_excess,
            })
    return pd.DataFrame(rows)


def summarize_net_return(df: pd.DataFrame) -> None:
    print("\n=== 净收益核算（T+1建仓，扣完整回合成本%.3f%%）===" % (ROUND_TRIP_COST * 100))
    for window in HOLD_WINDOWS:
        sub = df[df["window"] == window]["net_excess"].dropna()
        if sub.empty:
            print(f"\n持有{window}个交易日：无有效数据")
            continue
        mean = sub.mean()
        same_sign = (np.sign(sub) == np.sign(mean)).mean() if mean != 0 else 0.0
        t_stat, p_val = stats.ttest_1samp(sub, 0)
        print(f"持有{window}个交易日：净超额收益均值={mean:+.4%}  "
              f"同向占比={same_sign:.1%}  n={len(sub)}事件  t={t_stat:.2f}  p={p_val:.3f}")


def limit_up_check(events: pd.DataFrame, trade_days: pd.DatetimeIndex) -> pd.DataFrame:
    """检查T+1建仓日涨停不可执行的事件比例"""
    rows = []
    for _, ev in events.iterrows():
        entry_date = compute_entry_date(trade_days, ev["ann_date"])
        if entry_date is None:
            continue
        rows.append({
            "ts_code": ev["ts_code"], "entry_date": entry_date.date(),
            "blocked": limit_up_blocked(ev["ts_code"], entry_date),
        })
    return pd.DataFrame(rows)


def summarize_limit_up(df: pd.DataFrame) -> float:
    print("\n=== 涨跌停可执行性检查（T+1建仓日）===")
    blocked_pct = df["blocked"].mean()
    print(f"总事件数={len(df)}  涨停不可执行数={df['blocked'].sum()}  "
          f"不可执行比例={blocked_pct:.1%}")
    return blocked_pct


def capacity_estimate(events: pd.DataFrame, trade_days: pd.DatetimeIndex) -> float:
    """
    估算单次事件容量：建仓当日成交额的10%（行业惯用经验阈值）。
    注意：这不是"N只股票同时建仓的组合容量"（不同于指数调整事件，增持公告
    是随时间连续到达的事件流，不是固定批次），这里只估算单次事件能承载的
    资金量，供后续判断信号频率×单次容量的年化资金承载量级。
    """
    caps = []
    for _, ev in events.iterrows():
        entry_date = compute_entry_date(trade_days, ev["ann_date"])
        if entry_date is None:
            continue
        s = load_daily_amount(ev["ts_code"])
        s = s.loc[entry_date:entry_date]
        if not s.empty:
            caps.append(s.iloc[0] * 1000 * 0.10)  # amount单位千元，转元后取10%
    if not caps:
        return np.nan
    return float(np.mean(caps))


def main():
    pro = init_pro()
    pool_codes = load_pool_codes()
    print(f"股票池（沪深300+中证500并集）：{len(pool_codes)} 只")

    raw = load_increase_events_raw()
    print(f"增持公告原始记录（in_de='IN'，已限制股票池）："
          f"{len(raw[raw['ts_code'].isin(pool_codes)])} 条")

    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20160101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))

    events = dedup_to_events(raw, pool_codes, trade_days)
    print(f"去重合并后事件数：{len(events)}（合并窗口={MERGE_WINDOW_TRADING_DAYS}个交易日）")

    index_close = load_index_daily(pro, BENCHMARK_CODE)

    net_df = net_return_analysis(events, trade_days, index_close)
    summarize_net_return(net_df)

    limit_df = limit_up_check(events, trade_days)
    blocked_pct = summarize_limit_up(limit_df)

    avg_capacity = capacity_estimate(events, trade_days)
    print(f"\n单次事件容量估算（单只不超日均成交额10%）：约 {avg_capacity / 1e4:.1f} 万元" if pd.notna(avg_capacity) else "\n容量估算：无有效数据")

    out_dir = pathlib.Path(__file__).parent / "results" / "event_holder_increase"
    out_dir.mkdir(parents=True, exist_ok=True)
    net_df.to_csv(out_dir / "net_return_summary.csv", index=False)
    limit_df.to_csv(out_dir / "limit_up_check.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    print(f"\n结果已保存：{out_dir}")


if __name__ == "__main__":
    main()
