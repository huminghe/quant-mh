"""
财报披露时点相对排名效应（第十七轮候选④，事件研究，仅测核心前提）

假设：管理层倾向延迟披露坏消息、提前披露好消息（学术界"披露时机效应"，
Earnings Announcement Timing文献），预期同一报告期内更早披露的公司组，
披露后未来收益显著高于更晚披露的公司组。不预设方向由数据检验（若反向
显著，即"早披露反而跑输"，同样是有效信号，只是符号相反）。

数据与方法：
- 数据源：disclosure_date.parquet（fetch_disclosure_date.py拉取），用
  actual_date（实际披露日）而非end_date（报告期）作为PIT可知时点。
- 分组：按"距法定披露截止日天数"（一季报4/30，半年报8/31，三季报10/31，
  年报次年4/30）用固定阈值分5组（Q1=最早披露/离截止日最远，Q5=最晚披露/
  离截止日最近或逾期）。**不用**同报告期内的百分位排名分组——排名边界
  依赖该期内最晚披露公司的actual_date，而这在早披露公司买入时点根本
  还没发生，会造成前视偏差。固定阈值分组只用披露当天已知的信息，PIT安全。
- T+1建仓：actual_date（若非交易日先顺延）之后再顺延一个交易日买入，
  代表"披露发布后市场能够反应的最早可交易时点"。
- 测试窗口：20/60个交易日累计超额收益（先测短窗口，同event_holder_
  increase.py"先测前提"方法论）。
- 基准：中证800（000906.SH），覆盖沪深300+中证500全部成分股。
- 显著性检验：以"报告期"为样本单位（不是以个股为样本单位，避免同期
  内同ts_code跨期出现导致的样本非独立性被高估），每期算Q1组与Q5组
  平均超额收益的差值，对这个差值序列做单样本t检验（是否显著不为0）。

复用 event_index_rebalance.py 的通用工具函数（T+1建仓价/退出价、
基准窗口收益、成本口径、涨跌停可执行性检查、容量估算），避免重复
实现（DRY）。

**2026-08-24更新（用户决策）**：合并池（沪深300+中证500）60日窗口整体
显著(p=0.0057)，但拆开看主要由中证500驱动（p=0.0016，同向占比68.3%），
沪深300单独不显著（p=0.0885，同向占比56.1%）。用户决定只保留中证500，
做完整组合回测（涨跌停可执行性+容量估算+年化贡献量级），不纳入沪深300。
默认股票池改为仅中证500。

**2026-08-24修正（前视偏差）**：发现原分组方法（同报告期内按actual_date
百分位排名）有前视偏差，改为固定阈值分组（距法定截止日天数）后重新验证：
中证500 60日窗口价差均值(gross)+0.0242，扣成本+0.0225，t=+3.508，
**p=0.0011（比修正前更显著）**，同向占比73.2%（比修正前68.3%更高）。
20日窗口p=0.0538，仍不显著。信号在去除前视偏差后依然稳健。

用法：
  cd a_stock/backtest
  python event_disclosure_timing.py                # 默认仅中证500
  python event_disclosure_timing.py --index all    # 沪深300+中证500（历史对比用）
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import DATA_DIR
from fetch_disclosure_date import load_disclosure_date
from event_index_rebalance import (
    shift_trading_day, entry_price, exit_price,
    load_index_daily, index_window_return, ROUND_TRIP_COST,
    limit_up_blocked, load_daily_amount,
)

MEMBERS_FILES_ALL = [DATA_DIR / "hs300_members.parquet", DATA_DIR / "hs500_members.parquet"]
MEMBERS_FILES_HS500 = [DATA_DIR / "hs500_members.parquet"]
BENCHMARK_CODE = "000906.SH"  # 中证800，覆盖沪深300+中证500全部成分股

N_GROUPS = 5
HOLD_WINDOWS = [20, 60]  # 持有期（交易日）
MIN_STOCKS_PER_PERIOD = 50
EVENTS_PER_YEAR = 4  # 季报披露，每年4次（一/二/三/四季度）


def load_pool_codes(members_files: list = None) -> set:
    files = members_files if members_files is not None else MEMBERS_FILES_ALL
    codes = set()
    for f in files:
        if f.exists():
            m = pd.read_parquet(f)
            codes.update(m["con_code"].unique())
    return codes


def load_trade_days(pro_index_close: pd.Series) -> pd.DatetimeIndex:
    return pro_index_close.index


def _disclosure_deadline(end_date: pd.Timestamp) -> pd.Timestamp:
    """法定披露截止日（证监会规定：一季报4/30，半年报8/31，三季报10/31，年报次年4/30）"""
    m, y = end_date.month, end_date.year
    if m == 3:
        return pd.Timestamp(y, 4, 30)
    if m == 6:
        return pd.Timestamp(y, 8, 31)
    if m == 9:
        return pd.Timestamp(y, 10, 31)
    if m == 12:
        return pd.Timestamp(y + 1, 4, 30)
    return pd.NaT


# 固定分箱阈值（距法定截止日天数，披露当天即可算出，非同期排名——避免用
# 该报告期内"最晚披露公司"才产生的排名边界，那会造成前视偏差：早披露的
# 公司在披露当天根本不知道自己最终会被排进Q1还是Q2，因为排名依赖当期
# 尚未发生的其他公司的披露行为）
DEADLINE_BINS = [-10_000, 1, 3, 5, 10, 10_000]  # 升序分箱，对应days_to_deadline
DEADLINE_LABELS = ["Q5", "Q4", "Q3", "Q2", "Q1"]  # 分箱升序=披露越晚->越早，故标签降序（Q1=最早披露）


def assign_groups_per_period(disclosure: pd.DataFrame, pool_codes: set) -> pd.DataFrame:
    """
    按"距法定披露截止日天数"用固定阈值分N_GROUPS组（PIT安全：分组阈值是
    披露前就已知的固定常量，不依赖同期其他公司的披露时点）。
    返回：ts_code, end_date, actual_date, group（"Q1"~"Q5"）
    """
    sub = disclosure[disclosure["ts_code"].isin(pool_codes)].dropna(subset=["actual_date"]).copy()
    sub["deadline"] = sub["end_date"].apply(_disclosure_deadline)
    sub["days_to_deadline"] = (sub["deadline"] - sub["actual_date"]).dt.days
    # pd.cut要求单调递增边界；days_to_deadline越大=披露越早，DEADLINE_LABELS
    # 按边界升序排列对应"最晚->最早"，即Q5,Q4,Q3,Q2,Q1
    sub["group"] = pd.cut(sub["days_to_deadline"], bins=DEADLINE_BINS, labels=DEADLINE_LABELS)

    records = []
    for end_date, g in sub.groupby("end_date"):
        if len(g) < MIN_STOCKS_PER_PERIOD:
            continue
        records.append(g[["ts_code", "end_date", "actual_date", "group"]])

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def compute_group_excess_returns(
    grouped: pd.DataFrame, trade_days: pd.DatetimeIndex, index_close: pd.Series, hold_window: int,
) -> pd.DataFrame:
    """
    对每个报告期*分组，算T+1建仓、持有hold_window个交易日的等权组合超额收益。
    返回：index=end_date，columns=Q1..Q5，值=组合超额收益（gross，未扣成本，
    成本在Q1-Q5价差层面统一扣一次即可，因为两组都要经历一次完整回合）
    """
    rows = []
    for end_date, period_g in grouped.groupby("end_date"):
        row = {"end_date": end_date}
        for group_label, g in period_g.groupby("group"):
            rets = []
            for _, r in g.iterrows():
                ann = r["actual_date"]
                pos = trade_days.searchsorted(ann)
                if pos >= len(trade_days):
                    continue
                entry_date = shift_trading_day(trade_days, trade_days[pos], 1)
                if entry_date is None:
                    continue
                exit_date = shift_trading_day(trade_days, entry_date, hold_window)
                if exit_date is None:
                    continue
                p_in = entry_price(r["ts_code"], entry_date)
                p_out = exit_price(r["ts_code"], exit_date)
                if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
                    continue
                idx_ret = index_window_return(index_close, entry_date, exit_date)
                if pd.isna(idx_ret):
                    continue
                stock_ret = p_out / p_in - 1
                rets.append(stock_ret - idx_ret)
            row[group_label] = np.mean(rets) if rets else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("end_date")


def summarize_spread(group_returns: pd.DataFrame, hold_window: int) -> pd.Series:
    labels = [f"Q{j+1}" for j in range(N_GROUPS)]
    print(f"\n持有期 {hold_window} 个交易日，各期各组平均超额收益（gross）：")
    print(group_returns[labels].mean().apply(lambda v: f"{v:+.4f}"))

    spread = (group_returns["Q1"] - group_returns[f"Q{N_GROUPS}"]).dropna()
    net_spread = spread - ROUND_TRIP_COST  # Q1-Q5价差扣一次完整回合成本
    if len(spread) < 8:
        print(f"  样本报告期数={len(spread)}，样本量过少不做显著性检验")
        return spread
    t_stat, p_value = stats.ttest_1samp(spread, 0)
    same_sign = (np.sign(spread) == np.sign(spread.mean())).mean()
    print(f"  样本报告期数={len(spread)}")
    print(f"  Q1(最早披露)-Q{N_GROUPS}(最晚披露) 价差均值(gross)={spread.mean():+.4f}  "
          f"扣成本后={net_spread.mean():+.4f}")
    print(f"  t统计量={t_stat:+.3f}  p值={p_value:.4f}  同向占比={same_sign*100:.1f}%")
    passed = p_value < 0.05 and abs(net_spread.mean()) > 0
    print(f"  {'显著（p<0.05）' if passed else '不显著'}")
    return spread


# ── 涨跌停可执行性检查（Q1组，复用event_index_rebalance.py工具函数） ──────

def limit_up_check_q1(grouped: pd.DataFrame, trade_days: pd.DatetimeIndex) -> float:
    q1 = grouped[grouped["group"] == "Q1"]
    n_blocked, n_total = 0, 0
    for _, r in q1.iterrows():
        ann = r["actual_date"]
        pos = trade_days.searchsorted(ann)
        if pos >= len(trade_days):
            continue
        entry_date = shift_trading_day(trade_days, trade_days[pos], 1)
        if entry_date is None:
            continue
        n_total += 1
        if limit_up_blocked(r["ts_code"], entry_date):
            n_blocked += 1
    return n_blocked / n_total if n_total else float("nan")


# ── 容量估算（Q1组做多头，按event_index_rebalance.py同一经验口径） ─────

def capacity_estimate(grouped: pd.DataFrame, trade_days: pd.DatetimeIndex, hold_window: int) -> float:
    """Q1组个股在建仓窗口内平均日成交额，按单只不超过其自身日均成交额10%估算容量"""
    q1 = grouped[grouped["group"] == "Q1"]
    daily_amounts = []
    for _, r in q1.iterrows():
        ann = r["actual_date"]
        pos = trade_days.searchsorted(ann)
        if pos >= len(trade_days):
            continue
        entry_date = shift_trading_day(trade_days, trade_days[pos], 1)
        exit_date = shift_trading_day(trade_days, entry_date, hold_window) if entry_date is not None else None
        if entry_date is None or exit_date is None:
            continue
        s = load_daily_amount(r["ts_code"])
        s = s.loc[entry_date:exit_date]
        if not s.empty:
            daily_amounts.append(s.mean())
    if not daily_amounts:
        return float("nan")
    avg_daily_amount_yuan = float(np.mean(daily_amounts)) * 1000  # tushare amount单位千元
    n_stocks_per_period = q1.groupby("end_date").size().mean()
    return avg_daily_amount_yuan * 0.10 * n_stocks_per_period


def main():
    parser = argparse.ArgumentParser(description="财报披露时点相对排名效应事件研究")
    parser.add_argument("--index", choices=["hs500", "all"], default="hs500",
                        help="默认仅中证500（用户2026-08-24决策：沪深300子样本不显著，不纳入）")
    args = parser.parse_args()

    print("加载财报披露日期数据...")
    disclosure = load_disclosure_date()
    members_files = MEMBERS_FILES_HS500 if args.index == "hs500" else MEMBERS_FILES_ALL
    pool_codes = load_pool_codes(members_files)
    pool_desc = "中证500" if args.index == "hs500" else "沪深300+中证500历史成分股并集"
    print(f"股票池 {len(pool_codes)} 只（{pool_desc}）")

    grouped = assign_groups_per_period(disclosure, pool_codes)
    if grouped.empty:
        print("无有效分组数据，退出")
        return
    n_periods = grouped["end_date"].nunique()
    print(f"有效报告期数：{n_periods}，分组后合计 {len(grouped)} 条个股*报告期记录\n")

    print("加载中证800基准指数...")
    from fetch_index_members import init_pro
    pro = init_pro()
    index_close = load_index_daily(pro, BENCHMARK_CODE)
    trade_days = index_close.index

    print(f"{'='*60}")
    print("涨跌停可执行性检查（Q1组T+1建仓）")
    blocked_pct = limit_up_check_q1(grouped, trade_days)
    print(f"  涨停无法买入占比：{blocked_pct*100:.1f}%")

    for hold_window in HOLD_WINDOWS:
        print(f"{'='*60}")
        group_returns = compute_group_excess_returns(grouped, trade_days, index_close, hold_window)
        spread = summarize_spread(group_returns, hold_window)

        if hold_window == HOLD_WINDOWS[-1]:
            capacity_yuan = capacity_estimate(grouped, trade_days, hold_window)
            print(f"\n  容量估算（Q1组，单只不超过日均成交额10%）：{capacity_yuan/1e8:.2f} 亿元/次")
            net_mean = (spread - ROUND_TRIP_COST).mean() if len(spread) else float("nan")
            annual_contribution = net_mean * EVENTS_PER_YEAR
            print(f"  年化贡献量级估算（净超额均值×{EVENTS_PER_YEAR}次/年）：{annual_contribution*100:+.2f}%/年"
                  f"（仅Q1组资金，非整个组合年化收益）")


if __name__ == "__main__":
    main()
