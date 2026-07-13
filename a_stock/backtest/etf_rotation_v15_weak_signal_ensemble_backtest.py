"""
第十二轮方向2（续）：ML弱信号集成 —— 组合层面回测

背景：`etf_rotation_v15_weak_signal_ensemble.py` 的IC检验显示集成信号
IC=+0.038，年度同向占比75%，达到项目排除阈值（|IC|>=0.03 且同向占比>=60%），
按惯例需进入组合层面验证。但IC检验已经发现一个警示信号：集成信号IC
（0.038）低于资金流单独信号IC（0.063），说明集成没有产生互补增量，
反而被crowding/vol_ratio两个更弱的信号拖累。本脚本验证这一怀疑在
组合层面是否成立：动量+集成信号 是否优于 动量+资金流单独信号，
以及是否优于当前上线基线（无任何弱信号修正）。

方法：参照 etf_rotation_v3b_crowding.py 的软过滤框架，用集成信号
（月度横截面排名）替代拥挤度分位数，同样用连续打折/软过滤两种方式，
和"只用资金流单独信号"做对照。
"""

import sys
import time
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro

INIT_CASH = 1_000_000
COMMISSION = 0.0001
SLIPPAGE = 0.0002
BENCHMARK = "510300.SH"
START_DATE = "2019-01-01"   # 与fund_share数据覆盖对齐
IS_RATIO = 0.8
MOMENTUM_WINDOW = 25
TOP_N = 3
RISK_VOL_WINDOW = 21
CORR_WINDOW = 60
CORR_HIST_WINDOW = 252


def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_risk_adj_momentum(close_matrix: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score(series.iloc[i - MOMENTUM_WINDOW: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close_matrix.index)


def calc_crowding(close: pd.DataFrame, corr_window: int = CORR_WINDOW,
                   hist_window: int = CORR_HIST_WINDOW) -> pd.DataFrame:
    codes = list(close.columns)
    ret = close.pct_change()
    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(corr_window, len(close.index)):
        ret_win = ret.iloc[i - corr_window: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(np.nanmean(corr_arr, axis=1), index=ret_win.columns)
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values
    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(hist_window + corr_window, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - hist_window: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna()
            c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def load_amount_matrix() -> pd.DataFrame:
    daily_dir = pathlib.Path(__file__).parent.parent / "data" / "daily"
    dfs = {}
    for f in daily_dir.glob("*.parquet"):
        code = f.stem
        df = pd.read_parquet(f, columns=["trade_date", "amount"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        dfs[code] = df["amount"]
    result = pd.DataFrame(dfs)
    min_valid_rows = result.shape[1] // 2
    valid_mask = result.notna().sum(axis=1) >= min_valid_rows
    last_valid = result[valid_mask].index[-1]
    return result[result.index <= last_valid]


def fetch_fund_share_all(pro, codes: list, start_date: str) -> pd.DataFrame:
    today = pd.Timestamp.today().strftime("%Y%m%d")
    frames = {}
    for code in codes:
        try:
            df = pro.fund_share(ts_code=code, start_date=start_date, end_date=today)
            if not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").set_index("trade_date")
                frames[code] = df["fd_share"].astype(float)
            time.sleep(0.2)
        except Exception as e:
            print(f"  {code} 失败: {e}")
    if not frames:
        return pd.DataFrame()
    matrix = pd.DataFrame(frames)
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index()


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
                  boost_signal: pd.DataFrame = None, boost_mode: str = "none",
                  soft_factor: float = 0.5, threshold: float = 0.7,
                  top_n: int = TOP_N, init_cash: float = INIT_CASH) -> pd.Series:
    """
    boost_signal: 月度横截面排名（0~1，越大越好），index为调仓日
    boost_mode: none / continuous(得分×排名) / soft(排名<threshold时得分×soft_factor)
    """
    cash = init_cash
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        if date not in rebal_set:
            continue

        day_scores = scores.loc[date].dropna().copy()

        if boost_mode != "none" and boost_signal is not None and date in boost_signal.index:
            day_boost = boost_signal.loc[date]
            for code in day_scores.index:
                if code not in day_boost.index or pd.isna(day_boost[code]):
                    continue
                b = day_boost[code]
                if boost_mode == "continuous":
                    day_scores[code] *= (0.5 + b)   # 中性0.5，避免信号全灭
                elif boost_mode == "soft":
                    if b < threshold:
                        day_scores[code] *= soft_factor

        pos_scores = day_scores[day_scores > 0].nlargest(top_n)
        target_codes = list(pos_scores.index)

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
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
            elif diff < -price * 100:
                sell_shares = int(-diff / price / 100) * 100
                if sell_shares > 0 and current_shares >= sell_shares:
                    cash += sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


def calc_stats(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd,
            "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0}


def main():
    print("加载价格与成交额数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    print("\n计算动量得分...")
    scores = calc_risk_adj_momentum(close_full)[valid_codes]
    scores = scores[scores.index >= START_DATE]
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("计算拥挤度信号...")
    crowding = calc_crowding(close_full[valid_codes])
    crowding = crowding[crowding.index >= START_DATE]

    print("计算成交量确认信号...")
    amount = load_amount_matrix()
    amount = amount[[c for c in valid_codes if c in amount.columns]]
    amount = amount[amount.index >= START_DATE]
    vol_ratio = (amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan))

    print("拉取ETF份额数据（资金流信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    # 构造月度信号：集成排名(0~1) 和 资金流单独排名(0~1，反向)
    ensemble_rank, flow_only_rank = {}, {}
    for d in rebal_dates:
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
        if ranks:
            ensemble_rank[d] = pd.concat(ranks, axis=1).mean(axis=1)

        flow_s = flow_d.dropna()
        if len(flow_s) >= 5:
            flow_only_rank[d] = 1 - flow_s.rank(pct=True)

    ensemble_df = pd.DataFrame(ensemble_rank).T
    flow_only_df = pd.DataFrame(flow_only_rank).T

    bench = close[BENCHMARK].dropna()

    print("\n" + "=" * 80)
    print("组合层面回测：动量基线 vs +集成信号(连续/软过滤) vs +资金流单独信号")
    print("=" * 80)

    configs = [
        ("基线（纯动量，当前上线）", None, "none", 0.5, 0.7),
        ("+集成信号 连续打折", ensemble_df, "continuous", 0.5, 0.7),
        ("+集成信号 软过滤(<0.5×0.5)", ensemble_df, "soft", 0.5, 0.5),
        ("+集成信号 软过滤(<0.3×0.3)", ensemble_df, "soft", 0.3, 0.3),
        ("+资金流单独信号 连续打折", flow_only_df, "continuous", 0.5, 0.7),
        ("+资金流单独信号 软过滤(<0.5×0.5)", flow_only_df, "soft", 0.5, 0.5),
    ]

    rows = []
    n_days = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]

    for label, boost, mode, soft_factor, threshold in configs:
        nav = run_backtest(close, scores, rebal_dates, boost_signal=boost,
                            boost_mode=mode, soft_factor=soft_factor, threshold=threshold)
        stats = calc_stats(nav)

        rebal_is = [d for d in rebal_dates if d < split_date]
        rebal_oos = [d for d in rebal_dates if d >= split_date]
        close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
        sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]
        nav_is = run_backtest(close_is, sc_is, rebal_is, boost_signal=boost,
                               boost_mode=mode, soft_factor=soft_factor, threshold=threshold)
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, boost_signal=boost,
                                boost_mode=mode, soft_factor=soft_factor, threshold=threshold)
        s_is, s_oos = calc_stats(nav_is), calc_stats(nav_oos)

        rows.append({"配置": label, "夏普": f"{stats['Sharpe']:.3f}",
                     "年化": f"{stats['CAGR']*100:.1f}%", "回撤": f"{stats['MaxDD']*100:.1f}%",
                     "IS夏普": f"{s_is['Sharpe']:.3f}", "OOS夏普": f"{s_oos['Sharpe']:.3f}"})

    print(pd.DataFrame(rows).set_index("配置").to_string())

    baseline_sharpe = float(rows[0]["夏普"])
    best_row = max(rows[1:], key=lambda r: float(r["夏普"]))
    delta = float(best_row["夏普"]) - baseline_sharpe
    print("\n" + "=" * 80)
    print(f"结论：最优弱信号叠加配置「{best_row['配置']}」夏普={best_row['夏普']}，"
          f"基线夏普={baseline_sharpe:.3f}，Δ={delta:+.3f}")
    if delta > 0.02:
        print("→ 有正贡献，需做滚动窗口稳健性检验（样本仅6.5年，IS/OOS切分噪声大，不能直接采信）。")
    else:
        print("→ 无显著贡献（Δ<=0.02），判定排除，不采用，维持纯动量基线。")
    print("=" * 80)

    # ── 滚动窗口稳健性检验：最优配置 vs 基线 ──────────────────

    if delta > 0.02:
        print("\n" + "=" * 80)
        print(f"稳健性检验：滚动2年窗口夏普 —— 「{best_row['配置']}」 vs 基线")
        print("（样本仅6.5年，2年窗口已是可用的最长稳健窗口）")
        print("=" * 80)

        best_boost, best_mode, best_soft, best_thresh = next(
            (b, m, sf, th) for (lbl, b, m, sf, th) in configs if lbl == best_row["配置"]
        )
        nav_base_full = run_backtest(close, scores, rebal_dates)
        nav_best_full = run_backtest(close, scores, rebal_dates, boost_signal=best_boost,
                                      boost_mode=best_mode, soft_factor=best_soft, threshold=best_thresh)

        def roll_sharpe(s):
            r = s.pct_change().dropna()
            return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        window_days = 252 * 2
        rolling_base, rolling_best = [], []
        for i in range(window_days, len(nav_base_full)):
            rolling_base.append((nav_base_full.index[i], roll_sharpe(nav_base_full.iloc[i - window_days: i])))
        for i in range(window_days, len(nav_best_full)):
            rolling_best.append((nav_best_full.index[i], roll_sharpe(nav_best_full.iloc[i - window_days: i])))

        rs_base = pd.Series(dict(rolling_base))
        rs_best = pd.Series(dict(rolling_best))
        common_idx = rs_base.index.intersection(rs_best.index)
        improvement = rs_best[common_idx] - rs_base[common_idx]

        print(f"滚动2年夏普均值：基线={rs_base.mean():.2f}，最优配置={rs_best.mean():.2f}")
        print(f"差值：均值={improvement.mean():+.3f}，std={improvement.std():.3f}，"
              f"最小={improvement.min():+.3f}，最大={improvement.max():+.3f}")
        neg_ratio = (improvement < 0).mean()
        print(f"最优配置劣于基线的滚动窗口占比：{neg_ratio:.1%}")

        if neg_ratio > 0.4 or improvement.std() > abs(improvement.mean()) * 2:
            print("\n→ 滚动窗口不稳健（劣于基线占比>40%或波动远大于均值），"
                  "全样本Δ大概率是少数窗口驱动，判定为过拟合痕迹，不建议采用。")
        else:
            print("\n→ 滚动窗口相对稳健，正贡献在多数窗口成立，可考虑小规模试用并持续监控。")
        print("=" * 80)


if __name__ == "__main__":
    main()
