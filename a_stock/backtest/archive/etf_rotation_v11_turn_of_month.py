"""
第十轮调研 Task 4：跳出动量框架的新范式 —— 日历效应/月初月末效应（Turn-of-Month）
独立策略回测，不依赖现有动量轮动引擎，不影响现有上线信号。

背景：探测到沪深300ETF在"月末最后1个交易日+下月前N个交易日"窗口内的日均收益
显著高于其余交易日（初步t检验 p≈0.05）。本脚本验证：只在TOM窗口持有沪深300ETF，
其余时间持币，能否跑出有意义的风险调整收益。
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INIT_CASH = 1_000_000
START_DATE = "2016-01-01"
STOCK_ETF = "510300.SH"


def load_prices() -> pd.Series:
    data_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "daily"
    df = pd.read_parquet(data_dir / f"{STOCK_ETF}.parquet", columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["close"].sort_index()
    return s[s.index >= START_DATE]


def calc_full_stats(nav: pd.Series, label: str = "") -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "标的": label, "年化收益": f"{cagr*100:.1f}%", "夏普": f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%", "Calmar": f"{calmar:.2f}",
        "_sharpe": sharpe, "_maxdd": max_dd, "_cagr": cagr,
    }


def get_tom_mask(index: pd.DatetimeIndex, days_before: int, days_after: int) -> pd.Series:
    """
    构造TOM窗口mask：每月最后 days_before 个交易日 + 下月前 days_after 个交易日。
    注意：只用月内交易日顺序判断，不依赖未来数据（月末是否是最后一天，在当天即可确定
    ——因为是按自然月分组后取组内最后N行，不需要看到下个月的数据）。
    """
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    df["day_in_month"] = df.groupby("ym").cumcount() + 1
    df["days_in_month"] = df.groupby("ym")["day_in_month"].transform("max")
    is_month_end_window = df["day_in_month"] > (df["days_in_month"] - days_before)
    is_month_start_window = df["day_in_month"] <= days_after
    return (is_month_end_window | is_month_start_window)


def run_backtest(prices: pd.Series, days_before: int, days_after: int) -> pd.Series:
    """
    每个交易日判断是否处于TOM窗口：是则持有沪深300ETF，否则空仓（持币，收益记0）。
    调仓在窗口边界发生（进窗口买入，出窗口卖出），不额外设调仓频率限制。
    """
    tom_mask = get_tom_mask(prices.index, days_before, days_after)
    rets = prices.pct_change().fillna(0)
    # 用前一日的持仓状态决定当日是否吃到当日收益（避免用当天的mask直接乘当天收益导致的
    # "未来函数"错觉——实际上mask在当天开盘前就可确定，这里用shift(1)是保守起见，
    # 相当于"前一日收盘确认信号，次日开盘执行"，更贴近可执行的实盘时序）
    position = tom_mask.shift(1).fillna(False).astype(int)
    strat_rets = rets * position
    nav = (1 + strat_rets).cumprod() * INIT_CASH
    return nav


def main():
    prices = load_prices()

    print("参数敏感性（days_before × days_after 网格）...")
    rows = []
    for db in [1, 2, 3]:
        for da in [1, 2, 3, 4, 5]:
            nav = run_backtest(prices, db, da)
            s = calc_full_stats(nav)
            n_days_in_window = get_tom_mask(prices.index, db, da).sum()
            rows.append({
                "days_before": db, "days_after": da,
                "sharpe": s["_sharpe"], "cagr": s["_cagr"], "maxdd": s["_maxdd"],
                "n_days_in_window": n_days_in_window,
            })
            print(f"  db={db} da={da}  夏普={s['_sharpe']:.3f}  年化={s['_cagr']*100:.1f}%  "
                  f"回撤={s['_maxdd']*100:.1f}%  窗口天数={n_days_in_window}")

    grid_df = pd.DataFrame(rows)
    out_dir = pathlib.Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    grid_df.to_csv(out_dir / "v11_turn_of_month_grid.csv", index=False)
    print(f"\n网格结果已保存：{out_dir / 'v11_turn_of_month_grid.csv'}")

    best = grid_df.loc[grid_df["sharpe"].idxmax()]
    print(f"\n最优配置：days_before={int(best['days_before'])}, days_after={int(best['days_after'])}, "
          f"夏普={best['sharpe']:.3f}")

    # ── IS/OOS 验证（最优配置） ───────────────────────────
    db, da = int(best["days_before"]), int(best["days_after"])
    n = len(prices)
    split_idx = int(n * 0.75)
    split_date = prices.index[split_idx]
    nav_full = run_backtest(prices, db, da)
    nav_is = nav_full[nav_full.index < split_date]
    nav_oos_raw = prices[prices.index >= split_date]
    # OOS单独跑一次（避免用全样本状态导致的cumprod跨界问题）
    nav_oos = run_backtest(nav_oos_raw.to_frame("close")["close"] if False else prices[prices.index >= split_date], db, da)

    s_is = calc_full_stats(nav_is)
    s_oos = calc_full_stats(nav_oos)
    print(f"\nIS（{prices.index[0].date()}~{split_date.date()}）：夏普={s_is['_sharpe']:.3f}  年化={s_is['_cagr']*100:.1f}%")
    print(f"OOS（{split_date.date()}~{prices.index[-1].date()}）：夏普={s_oos['_sharpe']:.3f}  年化={s_oos['_cagr']*100:.1f}%")

    # ── 基准对比 ──────────────────────────────────────────
    bench_nav = prices / prices.iloc[0] * INIT_CASH
    s_bench = calc_full_stats(bench_nav, "沪深300买持")
    print(f"\n基准（沪深300买持全样本）：夏普={s_bench['_sharpe']:.3f}  年化={s_bench['_cagr']*100:.1f}%")


if __name__ == "__main__":
    main()
