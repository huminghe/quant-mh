"""
第十轮调研 Task 4：跳出动量框架的新范式 —— 股债收益差轮动
独立策略回测，不依赖现有动量轮动引擎，不影响现有上线信号。

信号：沪深300 E/P（PE_TTM倒数）− 中国10年期国债收益率 = 股债收益差。
数值越高代表股票相对债券越有性价比，用历史分位数打时机：
  - 收益差处于历史高分位（股票便宜）→ 持有沪深300ETF
  - 收益差处于历史低分位（股票贵）→ 持有国债ETF
数据来源：
  - 沪深300 PE_TTM：tushare pro.index_dailybasic，2016年至今全量可用（此前已验证）。
  - 中国10年期国债收益率：akshare ak.bond_zh_us_rate，2016-01-04至今可用。
  - 标的价格：本地 a_stock/data/daily/510300.SH.parquet、511010.SH.parquet。
"""

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import init_pro

INIT_CASH = 1_000_000
START_DATE = "2016-01-01"
STOCK_ETF = "510300.SH"
BOND_ETF = "511010.SH"
BENCHMARK_INDEX = "000300.SH"

PERCENTILE_WINDOW = 756  # 分位数回看窗口（约3年交易日），避免用未来数据定义"历史高低"
BUY_PCT = 0.5            # 收益差高于历史BUY_PCT分位 → 持股；否则持债


def load_prices() -> pd.DataFrame:
    data_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "daily"
    frames = {}
    for code in [STOCK_ETF, BOND_ETF]:
        df = pd.read_parquet(data_dir / f"{code}.parquet", columns=["trade_date", "close"])
        frames[code] = df.set_index("trade_date")["close"]
    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def load_yield_gap() -> pd.Series:
    """构造股债收益差 = 沪深300 E/P − 10年期国债收益率（单位：%）"""
    print("加载沪深300 PE_TTM...")
    pro = init_pro()
    pe_df = pro.index_dailybasic(
        ts_code=BENCHMARK_INDEX, start_date="20150101", end_date="20261231",
        fields="trade_date,pe_ttm",
    )
    pe_df["trade_date"] = pd.to_datetime(pe_df["trade_date"])
    pe = pe_df.sort_values("trade_date").set_index("trade_date")["pe_ttm"]
    ep = 1.0 / pe * 100  # E/P，百分比口径，与国债收益率一致

    print("加载10年期国债收益率...")
    import akshare as ak
    bond_df = ak.bond_zh_us_rate(start_date="20150101")
    bond_df["日期"] = pd.to_datetime(bond_df["日期"])
    bond_yield = bond_df.sort_values("日期").set_index("日期")["中国国债收益率10年"]

    gap = (ep - bond_yield).dropna()
    print(f"股债收益差数据范围：{gap.index[0].date()} ~ {gap.index[-1].date()}，{len(gap)} 条")
    return gap


def calc_full_stats(nav: pd.Series, label: str = "") -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    return {
        "标的": label, "年化收益": f"{cagr*100:.1f}%", "夏普": f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%", "Calmar": f"{calmar:.2f}", "月胜率": f"{win_rate:.1%}",
        "_sharpe": sharpe, "_maxdd": max_dd, "_cagr": cagr,
    }


def run_backtest(prices: pd.DataFrame, signal: pd.Series, buy_pct: float,
                  pct_window: int) -> pd.Series:
    """
    月度调仓：信号高于其历史 pct_window 日窗口内的 buy_pct 分位 → 持股，否则持债。
    分位数用滚动窗口（仅用截至当日的历史数据），避免用全样本分位数造成前视偏差。
    """
    idx = prices.index
    rolling_pct = signal.reindex(idx).ffill().rolling(pct_window, min_periods=pct_window // 2)
    threshold = rolling_pct.quantile(buy_pct)
    hold_stock = (signal.reindex(idx).ffill() > threshold).fillna(False)

    df = pd.DataFrame(index=idx)
    df["ym"] = df.index.to_period("M")
    rebal_dates = df.groupby("ym").apply(lambda x: x.index[0]).sort_index().tolist()
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    rebal_set = set(rebal_dates)

    cash = INIT_CASH
    code_held = None
    shares = 0.0
    nav_series = pd.Series(index=idx, dtype=float)

    for date in idx:
        if date < pd.Timestamp(START_DATE):
            nav_series[date] = np.nan
            continue
        pv = cash if code_held is None else shares * prices.loc[date, code_held]
        nav_series[date] = pv
        if date not in rebal_set:
            continue
        target = STOCK_ETF if hold_stock.loc[date] else BOND_ETF
        if target != code_held:
            cash = pv
            code_held = target
            shares = cash / prices.loc[date, code_held]
            cash = 0.0

    return nav_series.dropna()


def main():
    prices = load_prices()
    gap = load_yield_gap()

    nav = run_backtest(prices, gap, BUY_PCT, PERCENTILE_WINDOW)
    s = calc_full_stats(nav, "股债收益差轮动")
    print(f"\n股债收益差轮动：{s}")

    bench = prices[STOCK_ETF].reindex(nav.index)
    bench_nav = bench / bench.iloc[0] * INIT_CASH
    s_bench = calc_full_stats(bench_nav, "沪深300买持")
    print(f"基准（沪深300买持）：{s_bench}")

    bond_nav = prices[BOND_ETF].reindex(nav.index)
    bond_nav = bond_nav / bond_nav.iloc[0] * INIT_CASH
    s_bond = calc_full_stats(bond_nav, "国债ETF买持")
    print(f"对照（国债ETF买持）：{s_bond}")

    # ── 参数敏感性：buy_pct 网格 ──────────────────────────
    print("\n参数敏感性（buy_pct网格）...")
    rows = []
    for bp in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n = run_backtest(prices, gap, bp, PERCENTILE_WINDOW)
        st = calc_full_stats(n)
        rows.append({"buy_pct": bp, "夏普": st["_sharpe"], "年化": st["_cagr"],
                      "回撤": st["_maxdd"], "调仓次数": (n.diff() != 0).sum()})
        print(f"  buy_pct={bp}  夏普={st['_sharpe']:.3f}  年化={st['_cagr']*100:.1f}%  回撤={st['_maxdd']*100:.1f}%")

    out_dir = pathlib.Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "v11_stock_bond_yield_gap_grid.csv", index=False)
    print(f"\n网格结果已保存：{out_dir / 'v11_stock_bond_yield_gap_grid.csv'}")


if __name__ == "__main__":
    main()
