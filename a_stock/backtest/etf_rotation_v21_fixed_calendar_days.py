"""
调仓日期对比：仍是月度调仓（每月1次），但固定在每月第几个日历日
（1/5/10/15/20/25日）执行，对比这6种"锚定日"之间的效果差异。

用户澄清：不是把调仓频率从每月1次提到每月6次，而是月度调仓本身，
只是把"每月第一个交易日"这个锚点换成1/5/10/15/20/25日中的某一天
（当天非交易日则顺延至下一个交易日）。当前线上`signal_today.py`用的
是"每月第一个交易日"，等价于本次的锚定日=1。

分别对线上版（纯风险调整动量）和集成版（动量+拥挤度/成交量确认/资金流
打折，详见signal_shadow_ensemble.py）两个版本测试，各自用其可用的全部
历史样本。

沿用项目既有回测简化假设：信号当日收盘价直接成交（不额外模拟T+1延迟执行），
与本项目此前全部ETF轮动回测脚本一致，非本次新引入的简化。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch_data import load_close_matrix, init_pro
from etf_rotation_v16_signal_combo_ablation import (
    calc_risk_adj_momentum, calc_crowding, load_amount_matrix, fetch_fund_share_all,
)

INIT_CASH = 1_000_000
COMMISSION = 0.0001
SLIPPAGE = 0.0002
ROUND_TRIP_COST = 0.00164
BENCHMARK = "510300.SH"
MOMENTUM_WINDOW = 25
TOP_N = 3
IS_RATIO = 0.8
ANCHOR_DAYS = [1, 5, 10, 15, 20, 25]


# ── 调仓日期生成：每月仍只调仓1次，锚定在指定日历日 ──────────

def get_rebalance_dates_anchor_day(index: pd.DatetimeIndex, anchor_day: int) -> list:
    """每个日历月内，取 anchor_day 当天或之后的第一个交易日作为本月唯一调仓日"""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    dates = []
    for ym, grp in df.groupby("ym"):
        target = ym.to_timestamp() + pd.Timedelta(days=anchor_day - 1)
        candidates = grp.index[grp.index >= target]
        if len(candidates) > 0:
            dates.append(candidates[0])
    return sorted(dates)


# ── 回测引擎（复用v16的boost打折逻辑）────────────────────────

def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
                  boost_signal: pd.DataFrame = None, boost_mode: str = "none",
                  top_n: int = TOP_N, init_cash: float = INIT_CASH) -> tuple:
    cash = init_cash
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)
    turnover_events = []

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        if date not in rebal_set:
            continue

        day_scores = scores.loc[date].dropna().copy()

        if boost_mode == "continuous" and boost_signal is not None and date in boost_signal.index:
            day_boost = boost_signal.loc[date]
            for code in day_scores.index:
                if code in day_boost.index and not pd.isna(day_boost[code]):
                    day_scores[code] *= (0.5 + day_boost[code])

        pos_scores = day_scores[day_scores > 0].nlargest(top_n)
        target_codes = list(pos_scores.index)

        trade_value = 0.0
        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    sell_val = holdings[code] * price
                    cash += sell_val * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    trade_value += sell_val
                del holdings[code]

        if not target_codes:
            if port_value > 0:
                turnover_events.append(trade_value / port_value)
            continue

        n = len(target_codes)
        weights = {c: 1.0 / n for c in target_codes}

        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            buy_price = price * (1 + SLIPPAGE / 2)
            target_value = port_value * weights[code]
            current_shares = holdings.get(code, 0)
            current_value = current_shares * price
            diff = target_value - current_value
            if diff > buy_price * 100:
                buy_shares = int(diff / buy_price / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * buy_price * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = current_shares + buy_shares
                        trade_value += buy_shares * buy_price
            elif diff < -price * 100:
                sell_shares = int(-diff / price / 100) * 100
                if sell_shares > 0 and current_shares >= sell_shares:
                    cash += sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares
                    trade_value += sell_shares * price

        if port_value > 0:
            turnover_events.append(trade_value / port_value)

    meta = {}
    if turnover_events:
        n_rebal = len(turnover_events)
        avg_to = np.mean(turnover_events)
        years = (close.index[-1] - close.index[0]).days / 365.25
        meta["n_rebal"] = n_rebal
        meta["avg_turnover_per_rebal"] = avg_to
        meta["annual_turnover"] = avg_to * n_rebal / years
        meta["annual_cost_est"] = meta["annual_turnover"] * ROUND_TRIP_COST

    return nav_series.dropna(), meta


def calc_stats(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


def report(label: str, close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
           split_date: pd.Timestamp, boost_signal: pd.DataFrame = None, boost_mode: str = "none") -> dict:
    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]
    boost_is = boost_signal[boost_signal.index < split_date] if boost_signal is not None else None
    boost_oos = boost_signal[boost_signal.index >= split_date] if boost_signal is not None else None

    nav, meta = run_backtest(close, scores, rebal_dates, boost_signal, boost_mode)
    nav_is, _ = run_backtest(close_is, sc_is, rebal_is, boost_is, boost_mode)
    nav_oos, _ = run_backtest(close_oos, sc_oos, rebal_oos, boost_oos, boost_mode)

    stats = calc_stats(nav)
    stats_is = calc_stats(nav_is)
    stats_oos = calc_stats(nav_oos)
    decay = stats_oos["Sharpe"] / stats_is["Sharpe"] if stats_is["Sharpe"] > 0 else 0

    return {"label": label, "nav": nav, "stats": stats, "stats_is": stats_is,
            "stats_oos": stats_oos, "decay": decay, "meta": meta}


def combo_rank_at(d, crowding, vol_ratio, flow_1m):
    crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
    volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)
    flow_d = pd.Series(dtype=float)
    if not flow_1m.empty:
        idx = flow_1m.index[flow_1m.index <= d]
        if len(idx) > 0:
            flow_d = flow_1m.loc[idx[-1]]
    ranks = []
    for s, invert in [(crowd_d, True), (volr_d, False), (flow_d, True)]:
        s = s.dropna()
        if len(s) < 5:
            continue
        r = s.rank(pct=True)
        if invert:
            r = 1 - r
        ranks.append(r)
    if not ranks:
        return pd.Series(dtype=float)
    return pd.concat(ranks, axis=1).mean(axis=1)


def yearly_returns(nav: pd.Series) -> pd.Series:
    """按自然年切片算年度收益率（首尾用该年实际可用净值，不外推）"""
    yearly = {}
    for year, grp in nav.groupby(nav.index.year):
        if len(grp) < 2:
            continue
        yearly[year] = grp.iloc[-1] / grp.iloc[0] - 1
    return pd.Series(yearly)


def yearly_sharpe(nav: pd.Series) -> pd.Series:
    """按自然年切片算年度夏普（年内日收益年化，不满一年也直接用当年样本估计）"""
    yearly = {}
    for year, grp in nav.groupby(nav.index.year):
        rets = grp.pct_change().dropna()
        if len(rets) < 5 or rets.std() == 0:
            continue
        yearly[year] = rets.mean() / rets.std() * np.sqrt(252)
    return pd.Series(yearly)


def yearly_maxdd(nav: pd.Series) -> pd.Series:
    """按自然年切片算年度最大回撤（年内归一化，不跨年结转）"""
    yearly = {}
    for year, grp in nav.groupby(nav.index.year):
        if len(grp) < 2:
            continue
        rolling_max = grp.cummax()
        dd = (grp - rolling_max) / rolling_max
        yearly[year] = dd.min()
    return pd.Series(yearly)


def print_yearly_table(title: str, results: list):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    table = {r["label"]: yearly_returns(r["nav"]) for r in results}
    df = pd.DataFrame(table).sort_index()
    df_pct = df.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    print("\n【年度收益率】")
    print(df_pct.to_string())

    best_per_year = df.idxmax(axis=1)
    print("\n各年度收益率夺冠锚点：")
    print(best_per_year.to_string())

    table_sh = {r["label"]: yearly_sharpe(r["nav"]) for r in results}
    df_sh = pd.DataFrame(table_sh).sort_index()
    df_sh_fmt = df_sh.map(lambda x: f"{x:.2f}" if pd.notna(x) else "--")
    print("\n【年度夏普】")
    print(df_sh_fmt.to_string())

    table_dd = {r["label"]: yearly_maxdd(r["nav"]) for r in results}
    df_dd = pd.DataFrame(table_dd).sort_index()
    df_dd_fmt = df_dd.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--")
    print("\n【年度最大回撤】")
    print(df_dd_fmt.to_string())


def print_summary(title: str, results: list):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    rows = []
    for r in results:
        m = r["meta"]
        rows.append({
            "锚定日": r["label"], "年化": f"{r['stats']['CAGR']*100:.1f}%",
            "夏普": f"{r['stats']['Sharpe']:.3f}", "回撤": f"{r['stats']['MaxDD']*100:.1f}%",
            "IS夏普": f"{r['stats_is']['Sharpe']:.3f}", "OOS夏普": f"{r['stats_oos']['Sharpe']:.3f}",
            "OOS/IS": f"{r['decay']:.2f}",
            "调仓次数": m.get("n_rebal", "--"),
            "年化换手": f"{m.get('annual_turnover', 0)*100:.0f}%" if m else "--",
        })
    df = pd.DataFrame(rows).set_index("锚定日")
    print(df.to_string())

    sharpes = {r["label"]: r["stats"]["Sharpe"] for r in results}
    best = max(sharpes, key=sharpes.get)
    worst = min(sharpes, key=sharpes.get)
    print(f"\n夏普最高：{best}（{sharpes[best]:.3f}）  夏普最低：{worst}（{sharpes[worst]:.3f}）  "
          f"极差={sharpes[best]-sharpes[worst]:.3f}")


def main():
    print("加载价格与成交额数据...")
    close_full_raw = load_close_matrix()

    # ── Part 1：线上版，全历史2016-2026 ─────────────────────
    print("\n" + "=" * 90)
    print("Part 1：线上版（纯风险调整动量），锚定日=1/5/10/15/20/25，全历史")
    print("=" * 90)

    START_1 = "2016-01-01"
    close1 = close_full_raw[close_full_raw.index >= START_1]
    valid1 = [c for c in close1.columns if close1[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close1 = close1[valid1]
    print(f"有效标的：{len(valid1)} 只，{close1.index[0].date()} ~ {close1.index[-1].date()}")

    scores1 = calc_risk_adj_momentum(close_full_raw)[valid1]
    scores1 = scores1[scores1.index >= START_1]

    split_idx1 = int(len(close1) * IS_RATIO)
    split_date1 = close1.index[split_idx1]

    results1 = []
    for anchor in ANCHOR_DAYS:
        rebal = get_rebalance_dates_anchor_day(close1.index, anchor)
        r = report(f"每月{anchor}日", close1, scores1, rebal, split_date1)
        results1.append(r)
        print(f"  锚定日={anchor:>2}日  调仓次数={len(rebal):<4}  "
              f"夏普={r['stats']['Sharpe']:.3f}  年化={r['stats']['CAGR']*100:.1f}%  "
              f"回撤={r['stats']['MaxDD']*100:.1f}%")

    print_summary("Part 1 汇总：线上版，6个锚定日对比", results1)
    print_yearly_table("Part 1 分年度收益率：线上版，6个锚定日对比", results1)

    # ── Part 2：集成版，2019-2026 ─────────────────────────
    print("\n" + "=" * 90)
    print("Part 2：集成版（动量+拥挤度/成交量确认/资金流打折），锚定日=1/5/10/15/20/25，2019-2026")
    print("=" * 90)

    START_2 = "2019-01-01"
    close2 = close_full_raw[close_full_raw.index >= START_2]
    valid2 = [c for c in close2.columns if close2[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close2 = close2[valid2]
    print(f"有效标的：{len(valid2)} 只，{close2.index[0].date()} ~ {close2.index[-1].date()}")

    scores2 = calc_risk_adj_momentum(close_full_raw)[valid2]
    scores2 = scores2[scores2.index >= START_2]

    split_idx2 = int(len(close2) * IS_RATIO)
    split_date2 = close2.index[split_idx2]

    print("计算拥挤度信号（全历史滚动相关性，较慢）...")
    crowding = calc_crowding(close_full_raw[valid2])
    crowding = crowding[crowding.index >= START_2]

    print("计算成交量确认信号...")
    amount = load_amount_matrix()
    amount = amount[[c for c in valid2 if c in amount.columns]]
    amount = amount[amount.index >= START_2]
    vol_ratio = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取ETF份额数据（资金流信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid2, start_date=START_2.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    all_rebal2 = set()
    for anchor in ANCHOR_DAYS:
        all_rebal2 |= set(get_rebalance_dates_anchor_day(close2.index, anchor))
    boost_full = pd.DataFrame({d: combo_rank_at(d, crowding, vol_ratio, flow_1m) for d in sorted(all_rebal2)}).T

    results2 = []
    for anchor in ANCHOR_DAYS:
        rebal = get_rebalance_dates_anchor_day(close2.index, anchor)
        r = report(f"每月{anchor}日", close2, scores2, rebal, split_date2,
                   boost_signal=boost_full, boost_mode="continuous")
        results2.append(r)
        print(f"  锚定日={anchor:>2}日  调仓次数={len(rebal):<4}  "
              f"夏普={r['stats']['Sharpe']:.3f}  年化={r['stats']['CAGR']*100:.1f}%  "
              f"回撤={r['stats']['MaxDD']*100:.1f}%")

    print_summary("Part 2 汇总：集成版，6个锚定日对比", results2)
    print_yearly_table("Part 2 分年度收益率：集成版，6个锚定日对比", results2)

    print("\n" + "=" * 90)
    print("说明：所有配置调仓次数基本相同（均为约1次/月），差异纯粹来自"
          "'月内哪一天算分/成交'导致的动量得分取值不同，不涉及换手率差异。")
    print("=" * 90)


if __name__ == "__main__":
    main()
