"""
股东户数变化率因子组合回测（指数增强候选，已通过IC初筛）

背景：a_stock/docs/research.md「指数增强策略」章节候选因子。IC验证结果
（脚本不在本次任务范围内，结论由前置调研给出）：中证500全样本 ret20 IC均值
0.0518（p=0.0004）、ret60 IC均值0.0439（p=0.0003），2021-2025年度同向占比
83.3%（仅2026年至今约6个月转负，判断为风格切换扰动，非结构性失效，需持续
观察）。与动量因子相关性0.046、与SUE因子相关性0.044，均远低于0.5冗余阈值，
确认是独立增量信号。本脚本把该因子从"IC验证通过"推进到"组合回测"，参照
event_index_rebalance.py（T+1建仓/涨跌停检查/成本核算/容量估算）和
factor_multi_backtest_v2.py（月度调仓组合构建）两个既有框架的方法论。

因子定义：signal = -(holder_num_t / holder_num_{t-1} - 1)
即股东户数环比下降（筹码集中）信号越强。数据源 stk_holdernumber，
point-in-time用ann_date（只用截面当日已公告的最新两期记录）。

组合构建：
- 股票池：中证500历史成分股（本因子只在该指数通过IC验证，未在沪深300测试）
- 月度调仓：自然月末对齐到实际交易日（同factor_multi_backtest_v2.py的
  resample("ME")+trading-day-snap模式），每月末用截面signal选Top20%做多
  （A股不能融券做空个股，负向分支只能是"规避买入"规则，见下方
  bidirectional check，不构成组合净值曲线的一部分）
- 权重：等权（同factor_multi_backtest_v2.py惯例，避免市值加权引入的
  "大市值股票东户数天然更稳定"混淆因子暴露）
- T+1建仓：调仓日为月末交易日，实际建仓延后一个交易日（同
  event_index_rebalance.py的T+1约定），涨停无法买入的个股整月剔除
  （用限价单排队买入近似不现实，直接跳过更保守）
- 换手成本：不假设固定换手率（factor_multi_backtest_v2.py的turnover_est=0.5
  是历史遗留简化写法，本脚本按月度持仓集合实际变化算真实换手率——
  换手率=（新进个股数+退出个股数）/(2×组合规模)，再乘以项目级完整回合成本
  ROUND_TRIP_COST=0.00164（trading-standards.md唯一权威成本口径）
- 容量：调仓日个股日均成交额10%阈值（复用event_index_rebalance.py的
  capacity_estimate()方法论，适配为按月度持仓横截面平均而非离散事件窗口）
- 基准：中证500指数（000905.SH）
- 回测区间：2021-01 ~ 2026-08（与IC验证期一致）

用法：
  cd a_stock/backtest
  python factor_holder_number_backtest.py
  python factor_holder_number_backtest.py --top-pct 0.2
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import init_pro, load_close_panel, load_members_pit, DATA_DIR
from fetch_holder_number import load_holder_number, HOLDER_NUM_DIR

from event_index_rebalance import (
    shift_trading_day, limit_up_blocked, load_daily_amount,
    load_index_daily, index_window_return, ROUND_TRIP_COST,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_holder_number_backtest"

MEMBERS_FILE = DATA_DIR / "hs500_members.parquet"
INDEX_CODE   = "000905.SH"

START_DATE = "2021-01-01"
END_DATE   = "2026-08-31"

MIN_STOCKS_CROSS = 50   # 截面最少有效个股数（同factor_multi_backtest_v2.py惯例）
RISK_FREE_ANNUAL = 0.02


# ── PIT信号计算 ────────────────────────────────────────────

def compute_holder_signal_pit(ts_code: str, as_of: pd.Timestamp) -> float | None:
    """
    PIT计算单只股票的股东户数变化率信号：
    signal = -(holder_num_t / holder_num_{t-1} - 1)
    只用ann_date <= as_of的最新两条记录（按end_date排序取最后两期）。
    """
    df = load_holder_number(ts_code)
    if df.empty:
        return None
    valid = df[df["ann_date"] <= as_of].dropna(subset=["holder_num"])
    valid = valid.sort_values("end_date")
    if len(valid) < 2:
        return None
    prev, latest = valid.iloc[-2], valid.iloc[-1]
    if prev["holder_num"] <= 0:
        return None
    return -(latest["holder_num"] / prev["holder_num"] - 1)


def compute_cross_section_signal(codes: list[str], as_of: pd.Timestamp) -> pd.Series:
    vals = {}
    for code in codes:
        sig = compute_holder_signal_pit(code, as_of)
        if sig is not None and np.isfinite(sig):
            vals[code] = sig
    return pd.Series(vals)


# ── 绩效统计（同factor_multi_backtest_v2.py口径） ───────────

def sharpe(ret: pd.Series, freq: int = 12) -> float:
    if len(ret) < 2:
        return np.nan
    ann = ret.mean() * freq
    std = ret.std() * np.sqrt(freq)
    return np.nan if std < 1e-8 else (ann - RISK_FREE_ANNUAL) / std


def max_drawdown(nav: pd.Series) -> float:
    return ((nav - nav.cummax()) / nav.cummax()).min()


def annual_return(nav: pd.Series, freq: int = 12) -> float:
    n = len(nav)
    if n < 2:
        return np.nan
    return (1 + nav.iloc[-1] / nav.iloc[0] - 1) ** (1 / (n / freq)) - 1


def get_monthly_rebalance_dates(close_panel: pd.DataFrame) -> np.ndarray:
    """月末自然日对齐到实际交易日（同factor_multi_backtest_v2.py惯例）"""
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return monthly_last


# ── 组合回测主流程 ────────────────────────────────────────

def run_backtest(close_panel: pd.DataFrame, trade_days: pd.DatetimeIndex,
                  top_pct: float, index_close: pd.Series) -> tuple[pd.DataFrame, list[set]]:
    """
    月度调仓回测。每月末：
      1. 取当月末PIT成分股
      2. 算截面信号，选Top top_pct 做多
      3. T+1建仓（月末交易日的下一交易日），涨停不可执行的整月剔除
      4. 持有至下月对应建仓日，等权计算区间收益
    基准用中证500指数点位（index_ret，任务要求的主基准）；universe_ret
    （成分股等权）只作诊断参考，用于区分"选股alpha"和"等权vs市值加权指数
    结构性差异"两种不同来源的超额（不混入正式绩效指标）。
    返回：(月度收益明细DataFrame, 各期实际持仓集合列表)（后者用于换手率计算）
    """
    monthly_last = get_monthly_rebalance_dates(close_panel)

    records = []
    holdings_list = []  # 与records对齐，记录每期实际持仓（剔除涨停后）

    for i in range(len(monthly_last) - 1):
        month_end = pd.Timestamp(monthly_last[i])
        next_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=MEMBERS_FILE)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        signal = compute_cross_section_signal(available, month_end)
        if len(signal) < MIN_STOCKS_CROSS:
            continue

        n_select = max(1, int(len(signal) * top_pct))
        candidates = signal.nlargest(n_select).index.tolist()

        # T+1建仓：月末交易日的下一交易日
        entry_date = shift_trading_day(trade_days, month_end, 1)
        if entry_date is None:
            continue

        # 涨停不可执行的整月剔除（无法以当日价格买入）
        selected = [c for c in candidates if not limit_up_blocked(c, entry_date)]
        n_blocked = len(candidates) - len(selected)
        if not selected:
            continue

        ret_list = []
        for code in selected:
            p0 = close_panel[code].get(entry_date)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret_list.append(p1 / p0 - 1)
        if not ret_list:
            continue
        gross_ret = float(np.mean(ret_list))

        bm_rets = []
        for code in available:
            p0 = close_panel[code].get(entry_date)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        universe_ret = float(np.mean(bm_rets)) if bm_rets else np.nan
        index_ret = index_window_return(index_close, entry_date, next_end)

        records.append({
            "date": month_end, "entry_date": entry_date, "next_end": next_end,
            "gross_ret": gross_ret, "universe_ret": universe_ret, "index_ret": index_ret,
            "n_candidates": len(candidates), "n_selected": len(selected),
            "n_blocked": n_blocked,
        })
        holdings_list.append(set(selected))

    ret_df = pd.DataFrame(records).set_index("date") if records else pd.DataFrame()
    return ret_df, holdings_list


def compute_actual_turnover(holdings_list: list[set]) -> pd.Series:
    """
    实际换手率：连续两期持仓集合的差异比例。
    换手率 = (新进个股数 + 退出个股数) / (2 × 组合规模)
    第一期无上期持仓可比，换手率记为1.0（全新建仓）。
    """
    turnovers = []
    prev = None
    for cur in holdings_list:
        if prev is None:
            turnovers.append(1.0)
        else:
            n_in = len(cur - prev)
            n_out = len(prev - cur)
            denom = len(cur) + len(prev)
            turnovers.append((n_in + n_out) / denom if denom > 0 else 0.0)
        prev = cur
    return pd.Series(turnovers)


def apply_cost(ret_df: pd.DataFrame, holdings_list: list[set]) -> pd.DataFrame:
    """按实际换手率扣成本，得到net_ret列"""
    turnover = compute_actual_turnover(holdings_list)
    ret_df = ret_df.copy()
    ret_df["turnover"] = turnover.values
    ret_df["cost"] = ret_df["turnover"] * ROUND_TRIP_COST
    ret_df["net_ret"] = ret_df["gross_ret"] - ret_df["cost"]
    return ret_df


# ── 容量估算 ──────────────────────────────────────────────

def capacity_estimate(ret_df: pd.DataFrame, holdings_list: list[set]) -> float:
    """
    容量估算：各期实际建仓日个股日均成交额10%阈值，取全部持有期均值后
    按平均组合规模折算总容量（同event_index_rebalance.py方法论，适配为
    按月度持仓横截面平均而非离散事件窗口）。
    """
    daily_amounts = []
    sizes = []
    for (_, row), holdings in zip(ret_df.iterrows(), holdings_list):
        sizes.append(len(holdings))
        for code in holdings:
            s = load_daily_amount(code)
            s = s.loc[row["entry_date"]:row["entry_date"]]
            if not s.empty:
                daily_amounts.append(s.iloc[0])
    if not daily_amounts or not sizes:
        return np.nan
    avg_daily_amount_yuan = float(np.mean(daily_amounts)) * 1000  # 千元转元
    avg_size = float(np.mean(sizes))
    per_stock_capacity = avg_daily_amount_yuan * 0.10
    return per_stock_capacity * avg_size


# ── 反向信号"规避买入"检验 ────────────────────────────────

def run_avoid_buy_check(close_panel: pd.DataFrame, trade_days: pd.DatetimeIndex,
                         top_pct: float) -> pd.DataFrame:
    """
    反向检验：股东户数上升（signal最负）组是否显著跑输基准。
    这只能是"规避买入"规则（从候选池剔除），不能构造为组合净值曲线里的
    空头持仓——A股个股不能融券做空（a_stock/CLAUDE.md），本函数只输出
    该组的月度收益供对比，不参与任何净值/成本/容量计算。
    """
    monthly_last = get_monthly_rebalance_dates(close_panel)
    records = []
    for i in range(len(monthly_last) - 1):
        month_end = pd.Timestamp(monthly_last[i])
        next_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=MEMBERS_FILE)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        signal = compute_cross_section_signal(available, month_end)
        if len(signal) < MIN_STOCKS_CROSS:
            continue

        n_select = max(1, int(len(signal) * top_pct))
        worst_group = signal.nsmallest(n_select).index.tolist()  # 股东户数上升最多的组

        entry_date = shift_trading_day(trade_days, month_end, 1)
        if entry_date is None:
            continue

        ret_list, bm_rets = [], []
        for code in worst_group:
            p0 = close_panel[code].get(entry_date)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret_list.append(p1 / p0 - 1)
        for code in available:
            p0 = close_panel[code].get(entry_date)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        if not ret_list or not bm_rets:
            continue

        records.append({
            "date": month_end,
            "worst_group_ret": float(np.mean(ret_list)),
            "universe_ret": float(np.mean(bm_rets)),
        })
    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 结果输出 ──────────────────────────────────────────────

def summarize_performance(ret_df: pd.DataFrame) -> None:
    strat_gross = ret_df["gross_ret"]
    strat_net = ret_df["net_ret"]
    bench = ret_df["index_ret"]          # 主基准：中证500指数点位（任务要求）
    universe = ret_df["universe_ret"]    # 诊断参考：成分股等权

    nav_gross = (1 + strat_gross).cumprod()
    nav_net = (1 + strat_net).cumprod()
    nav_bench = (1 + bench).cumprod()

    excess_gross = strat_gross - bench
    excess_net = strat_net - bench
    excess_vs_universe = strat_net - universe  # 剔除等权结构性差异后的纯选股alpha

    print("\n=== 组合回测绩效（月度调仓，Top信号做多，等权）===")
    print(f"样本月数: {len(ret_df)}")
    print(f"策略年化收益（gross）: {annual_return(nav_gross)*100:.2f}%")
    print(f"策略年化收益（net，扣实际换手成本）: {annual_return(nav_net)*100:.2f}%")
    print(f"基准（中证500指数000905.SH）年化收益: {annual_return(nav_bench)*100:.2f}%")
    print(f"超额年化 vs 中证500指数（gross）: {excess_gross.mean()*12*100:.2f}%")
    print(f"超额年化 vs 中证500指数（net）: {excess_net.mean()*12*100:.2f}%")
    print(f"超额年化 vs 成分股等权universe（net，诊断参考——剔除等权/市值加权"
          f"结构性差异后的纯选股alpha）: {excess_vs_universe.mean()*12*100:.2f}%")
    print(f"策略夏普（net）: {sharpe(strat_net):.3f}")
    print(f"信息比率（net超额vs指数/超额波动，年化）: "
          f"{(excess_net.mean()*12) / (excess_net.std()*np.sqrt(12)) if excess_net.std() > 1e-8 else np.nan:.3f}")
    print(f"策略最大回撤（net）: {max_drawdown(nav_net)*100:.2f}%")
    print(f"月胜率（net超额vs指数>0）: {(excess_net > 0).mean()*100:.1f}%")
    print(f"平均月换手率: {ret_df['turnover'].mean()*100:.1f}%")
    print(f"平均月度成本拖累: {ret_df['cost'].mean()*100:.3f}%")
    print(f"平均涨停剔除比例: {(ret_df['n_blocked'] / ret_df['n_candidates']).mean()*100:.1f}%")


def summarize_annual(ret_df: pd.DataFrame) -> None:
    print("\n=== 年度分拆（vs 中证500指数，含2026年独立观察） ===")
    excess_net = ret_df["net_ret"] - ret_df["index_ret"]
    for y in sorted(ret_df.index.year.unique()):
        yr = excess_net[excess_net.index.year == y]
        gross_yr = (ret_df["gross_ret"] - ret_df["index_ret"])[excess_net.index.year == y]
        print(f"  {y}: net超额均值={yr.mean()*100:+.3f}%/月  "
              f"gross超额均值={gross_yr.mean()*100:+.3f}%/月  "
              f"同向占比(net>0)={(yr > 0).mean()*100:.0f}%  n={len(yr)}月")


def summarize_avoid_buy(avoid_df: pd.DataFrame) -> None:
    print("\n=== 反向信号检验：股东户数上升组“规避买入”效应 ===")
    print("（该组不构造为组合净值曲线，A股个股不可做空，仅作为选股候选池"
          "剔除规则的依据，此处只输出诊断统计）")
    if avoid_df.empty:
        print("  无有效数据")
        return
    excess = avoid_df["worst_group_ret"] - avoid_df["universe_ret"]
    from scipy import stats as scipy_stats
    t_stat, p_val = scipy_stats.ttest_1samp(excess.dropna(), 0)
    same_sign = (excess < 0).mean()  # 判断是否显著跑输universe（规避买入成立的方向）
    print(f"  样本月数: {len(avoid_df)}")
    print(f"  跑输universe月均值: {excess.mean()*100:+.3f}%/月")
    print(f"  跑输占比（该组收益<universe）: {same_sign*100:.1f}%")
    print(f"  t统计量={t_stat:.2f}  p={p_val:.4f}")
    if excess.mean() < 0 and p_val < 0.05:
        print("  结论：显著跑输，“规避买入”规则成立（应从候选池中剔除该组个股）")
    else:
        print("  结论：未达到显著跑输标准，“规避买入”规则不成立或证据不足")


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="股东户数变化率因子组合回测")
    parser.add_argument("--top-pct", type=float, default=0.2,
                        help="截面信号Top百分比做多（默认20%%）")
    args = parser.parse_args()

    if not MEMBERS_FILE.exists():
        print(f"缺少 {MEMBERS_FILE}，请先运行 fetch_index_members.py --index hs500")
        return
    if not HOLDER_NUM_DIR.exists():
        print(f"缺少 {HOLDER_NUM_DIR}，请先运行 fetch_holder_number.py --index hs500")
        return

    pro = init_pro()
    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20200101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))

    members = pd.read_parquet(MEMBERS_FILE)
    all_codes = sorted(members["con_code"].unique())
    print(f"中证500历史成分股：{len(all_codes)} 只")

    print("加载收盘价面板...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"  面板区间：{close_panel.index.min().date()} ~ {close_panel.index.max().date()}")

    print("加载中证500指数点位...")
    index_close = load_index_daily(pro, INDEX_CODE)

    print(f"\n开始月度调仓回测（Top{args.top_pct:.0%}做多，2021-01~2026-08）...")
    ret_df, holdings_list = run_backtest(close_panel, trade_days, args.top_pct, index_close)
    if ret_df.empty:
        print("回测无有效结果，退出")
        return
    ret_df = apply_cost(ret_df, holdings_list)

    summarize_performance(ret_df)
    summarize_annual(ret_df)

    print("\n开始容量估算...")
    capacity_yuan = capacity_estimate(ret_df, holdings_list)
    if pd.notna(capacity_yuan):
        print(f"容量估算（单只不超日均成交额10%，按平均组合规模折算）：约 {capacity_yuan / 1e8:.2f} 亿元")
    else:
        print("容量估算：无有效成交额数据")

    print("\n开始反向信号“规避买入”检验...")
    avoid_df = run_avoid_buy_check(close_panel, trade_days, args.top_pct)
    summarize_avoid_buy(avoid_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ret_df.to_csv(OUTPUT_DIR / "monthly_returns.csv")
    if not avoid_df.empty:
        avoid_df.to_csv(OUTPUT_DIR / "avoid_buy_check.csv")
    print(f"\n结果已保存：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
