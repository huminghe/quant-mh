"""
突显效应（ST）因子组合回测（指数增强候选，已通过IC初筛+冗余性检验）

背景：`factor_ic_salience.py` 验证ST因子（突显加权收益-等权收益，负向使用）
月度截面Rank IC：沪深300 IC均值+0.0356（ICIR 0.275，年度同向72.7%），中证500
IC均值+0.0363（ICIR 0.358，年度同向90.9%），均通过项目既定初筛阈值。冗余性
检验（剔除21日简单反转收益的排序成分后残差IC不降反升）确认ST不是简单反转
因子的马甲，有独立增量。但2025-2026年IC明显转弱转负（沪深3002025年-0.0351、
2026年-0.0452；中证500 2026年-0.0295），与第十三轮股东户数变化率因子"IC初筛
通过但组合回测暴露近两年连续衰减"的教训模式相似，必须做完整组合回测的年度
拆分才能下最终结论（第十三轮教训，不能止步于IC初筛）。

本脚本参照 factor_holder_number_backtest.py（T+1建仓/涨跌停检查/实际换手成本
/容量估算/年度拆分/反向"规避买入"检验）的完整方法论，把ST因子从"IC验证通过"
推进到"组合回测"。

因子定义：signal = -ST_t（ST_t见factor_ic_salience.py的
compute_salience_weighted_return，突显加权收益-等权收益，PIT用过去21个交易日
滚动窗口，不依赖未来数据）。

组合构建（与factor_holder_number_backtest.py完全一致的方法论）：
- 股票池：沪深300 + 中证500（两个指数都通过IC初筛，都测，看效应是否在两个
  指数上都能转化为组合层面alpha，呼应"机构持股比例"异质性问题）
- 月度调仓，Top20%等权做多，T+1建仓，涨停剔除，实际换手成本
  （ROUND_TRIP_COST=0.00164，trading-standards.md唯一权威成本口径）
- 基准：对应指数点位；成分股等权universe作诊断参考（剔除等权/市值加权
  结构性差异后的纯选股alpha）
- 回测区间：2016-01 ~ 2026-06（与IC验证期一致）
- 反向信号"规避买入"检验：ST最高（最显眼正收益）组是否显著跑输universe

用法：
  cd a_stock/backtest
  python factor_salience_backtest.py --index hs300
  python factor_salience_backtest.py --index hs500
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import init_pro, load_close_panel, load_members_pit, DATA_DIR  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from event_index_rebalance import (  # noqa: E402
    shift_trading_day, limit_up_blocked, load_daily_amount,
    load_index_daily, index_window_return, ROUND_TRIP_COST,
)
from factor_ic_salience import compute_salience_weighted_return, SALIENCE_WINDOW  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_salience_backtest"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_CROSS = 50
RISK_FREE_ANNUAL = 0.02

INDEX_CONFIG = {
    "hs300": {"name": "沪深300", "members_file": DATA_DIR / "hs300_members.parquet", "index_code": "000300.SH"},
    "hs500": {"name": "中证500", "members_file": DATA_DIR / "hs500_members.parquet", "index_code": "000905.SH"},
}


# ── PIT信号计算 ────────────────────────────────────────────

def compute_cross_section_signal(close_panel: pd.DataFrame, codes: list[str],
                                   month_end: pd.Timestamp) -> pd.Series:
    """
    signal = -ST_t（ST见factor_ic_salience.compute_salience_weighted_return）。
    只用month_end之前的历史数据（含当日），PIT安全。
    """
    daily_ret = close_panel[codes].pct_change()
    hist = daily_ret.loc[:month_end]
    if len(hist) < SALIENCE_WINDOW + 1:
        return pd.Series(dtype=float)
    window = hist.iloc[-SALIENCE_WINDOW:]
    st = compute_salience_weighted_return(window)
    return -st  # 负向使用：高ST（显眼正收益被高估）-> 低信号，回避


# ── 绩效统计（同factor_holder_number_backtest.py口径） ──────

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
                  top_pct: float, index_close: pd.Series, members_file: pathlib.Path
                  ) -> tuple[pd.DataFrame, list[set]]:
    monthly_last = get_monthly_rebalance_dates(close_panel)

    records = []
    holdings_list = []

    for i in range(len(monthly_last) - 1):
        month_end = pd.Timestamp(monthly_last[i])
        next_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        signal = compute_cross_section_signal(close_panel, available, month_end)
        signal = signal.dropna()
        if len(signal) < MIN_STOCKS_CROSS:
            continue

        n_select = max(1, int(len(signal) * top_pct))
        candidates = signal.nlargest(n_select).index.tolist()

        entry_date = shift_trading_day(trade_days, month_end, 1)
        if entry_date is None:
            continue

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
    turnover = compute_actual_turnover(holdings_list)
    ret_df = ret_df.copy()
    ret_df["turnover"] = turnover.values
    ret_df["cost"] = ret_df["turnover"] * ROUND_TRIP_COST
    ret_df["net_ret"] = ret_df["gross_ret"] - ret_df["cost"]
    return ret_df


def capacity_estimate(ret_df: pd.DataFrame, holdings_list: list[set]) -> float:
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
    avg_daily_amount_yuan = float(np.mean(daily_amounts)) * 1000
    avg_size = float(np.mean(sizes))
    per_stock_capacity = avg_daily_amount_yuan * 0.10
    return per_stock_capacity * avg_size


# ── 反向信号"规避买入"检验 ────────────────────────────────

def run_avoid_buy_check(close_panel: pd.DataFrame, trade_days: pd.DatetimeIndex,
                         top_pct: float, members_file: pathlib.Path) -> pd.DataFrame:
    """
    反向检验：ST最高（最显眼正收益，理论上被高估）组是否显著跑输universe。
    只能是"规避买入"规则，不构造为空头持仓（A股个股不可做空）。
    """
    monthly_last = get_monthly_rebalance_dates(close_panel)
    records = []
    for i in range(len(monthly_last) - 1):
        month_end = pd.Timestamp(monthly_last[i])
        next_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        signal = compute_cross_section_signal(close_panel, available, month_end)
        signal = signal.dropna()
        if len(signal) < MIN_STOCKS_CROSS:
            continue

        n_select = max(1, int(len(signal) * top_pct))
        worst_group = signal.nsmallest(n_select).index.tolist()  # signal最低=ST最高=最该回避

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

def summarize_performance(ret_df: pd.DataFrame, index_name: str, index_code: str) -> None:
    strat_gross = ret_df["gross_ret"]
    strat_net = ret_df["net_ret"]
    bench = ret_df["index_ret"]
    universe = ret_df["universe_ret"]

    nav_gross = (1 + strat_gross).cumprod()
    nav_net = (1 + strat_net).cumprod()
    nav_bench = (1 + bench).cumprod()

    excess_gross = strat_gross - bench
    excess_net = strat_net - bench
    excess_vs_universe = strat_net - universe

    print(f"\n=== 组合回测绩效（{index_name}，月度调仓，ST因子Top信号做多，等权） ===")
    print(f"样本月数: {len(ret_df)}")
    print(f"策略年化收益（gross）: {annual_return(nav_gross)*100:.2f}%")
    print(f"策略年化收益（net，扣实际换手成本）: {annual_return(nav_net)*100:.2f}%")
    print(f"基准（{index_name}指数{index_code}）年化收益: {annual_return(nav_bench)*100:.2f}%")
    print(f"超额年化 vs 指数（gross）: {excess_gross.mean()*12*100:.2f}%")
    print(f"超额年化 vs 指数（net）: {excess_net.mean()*12*100:.2f}%")
    print(f"超额年化 vs 成分股等权universe（net，诊断参考）: {excess_vs_universe.mean()*12*100:.2f}%")
    print(f"策略夏普（net）: {sharpe(strat_net):.3f}")
    print(f"信息比率（net超额vs指数/超额波动，年化）: "
          f"{(excess_net.mean()*12) / (excess_net.std()*np.sqrt(12)) if excess_net.std() > 1e-8 else np.nan:.3f}")
    print(f"策略最大回撤（net）: {max_drawdown(nav_net)*100:.2f}%")
    print(f"月胜率（net超额vs指数>0）: {(excess_net > 0).mean()*100:.1f}%")
    print(f"平均月换手率: {ret_df['turnover'].mean()*100:.1f}%")
    print(f"平均月度成本拖累: {ret_df['cost'].mean()*100:.3f}%")
    print(f"平均涨停剔除比例: {(ret_df['n_blocked'] / ret_df['n_candidates']).mean()*100:.1f}%")


def summarize_annual(ret_df: pd.DataFrame) -> None:
    print("\n=== 年度分拆（vs 指数，含2026年独立观察） ===")
    excess_net = ret_df["net_ret"] - ret_df["index_ret"]
    for y in sorted(ret_df.index.year.unique()):
        yr = excess_net[excess_net.index.year == y]
        gross_yr = (ret_df["gross_ret"] - ret_df["index_ret"])[excess_net.index.year == y]
        print(f"  {y}: net超额均值={yr.mean()*100:+.3f}%/月  "
              f"gross超额均值={gross_yr.mean()*100:+.3f}%/月  "
              f"同向占比(net>0)={(yr > 0).mean()*100:.0f}%  n={len(yr)}月")


def summarize_avoid_buy(avoid_df: pd.DataFrame) -> None:
    print("\n=== 反向信号检验：ST最高（最显眼正收益）组“规避买入”效应 ===")
    print("（该组不构造为组合净值曲线，A股个股不可做空，仅作为选股候选池"
          "剔除规则的依据，此处只输出诊断统计）")
    if avoid_df.empty:
        print("  无有效数据")
        return
    excess = avoid_df["worst_group_ret"] - avoid_df["universe_ret"]
    from scipy import stats as scipy_stats
    t_stat, p_val = scipy_stats.ttest_1samp(excess.dropna(), 0)
    same_sign = (excess < 0).mean()
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
    parser = argparse.ArgumentParser(description="突显效应ST因子组合回测")
    parser.add_argument("--index", choices=["hs300", "hs500"], required=True)
    parser.add_argument("--top-pct", type=float, default=0.2,
                        help="截面信号Top百分比做多（默认20%%）")
    args = parser.parse_args()

    cfg = INDEX_CONFIG[args.index]
    members_file = cfg["members_file"]
    index_name = cfg["name"]
    index_code = cfg["index_code"]

    if not members_file.exists():
        print(f"缺少 {members_file}")
        return

    pro = init_pro()
    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20150101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))

    members = pd.read_parquet(members_file)
    all_codes = sorted(members["con_code"].unique())
    print(f"{index_name}历史成分股：{len(all_codes)} 只")

    print("加载收盘价面板...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"  面板区间：{close_panel.index.min().date()} ~ {close_panel.index.max().date()}")

    print(f"加载{index_name}指数点位...")
    index_close = load_index_daily(pro, index_code)

    print(f"\n开始月度调仓回测（Top{args.top_pct:.0%}做多，2016-01~2026-06）...")
    ret_df, holdings_list = run_backtest(close_panel, trade_days, args.top_pct, index_close, members_file)
    if ret_df.empty:
        print("回测无有效结果，退出")
        return
    ret_df = apply_cost(ret_df, holdings_list)

    summarize_performance(ret_df, index_name, index_code)
    summarize_annual(ret_df)

    print("\n开始容量估算...")
    capacity_yuan = capacity_estimate(ret_df, holdings_list)
    if pd.notna(capacity_yuan):
        print(f"容量估算（单只不超日均成交额10%，按平均组合规模折算）：约 {capacity_yuan / 1e8:.2f} 亿元")
    else:
        print("容量估算：无有效成交额数据")

    print("\n开始反向信号“规避买入”检验...")
    avoid_df = run_avoid_buy_check(close_panel, trade_days, args.top_pct, members_file)
    summarize_avoid_buy(avoid_df)

    out_dir = OUTPUT_DIR / args.index
    out_dir.mkdir(parents=True, exist_ok=True)
    ret_df.to_csv(out_dir / "monthly_returns.csv")
    if not avoid_df.empty:
        avoid_df.to_csv(out_dir / "avoid_buy_check.csv")
    print(f"\n结果已保存：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
