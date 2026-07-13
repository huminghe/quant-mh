"""
第十二轮候选池严谨复核后的组合消融实验

背景：`etf_rotation_v15_weak_signal_ensemble_backtest.py` 只测试了
"三信号等权集成 vs 资金流单独" 两种组合方式，没有系统扫描全部子集。
本轮用Explore子agent对12轮调研全部信号重新核验分类后，确认严谨候选池
仅为3个横截面弱信号：拥挤度(crowding)、成交量确认(vol_ratio)、
资金流反向(flow)，其余候选（偏度、风格轮动、跨市场溢出、离散度regime）
均因方向不稳定/不可执行/共同信号性质被排除，不进入本次消融实验。

本脚本对这3个信号的全部非空子集（共7种：3单信号+3两两组合+1三者全部）
做等权排名集成，统一用"连续打折"叠加方式（v15验证中表现最优的方式），
比较组合层面夏普/年化/回撤，找出真正最优的信号子集。
"""

import sys
import itertools
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
START_DATE = "2019-01-01"
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
    import time
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
                    day_scores[code] *= (0.5 + b)
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
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


def roll_sharpe(s):
    r = s.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0


def main():
    print("加载价格与成交额数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    print("计算动量得分...")
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

    # 三个信号统一转为月度横截面排名（0~1，越大越好，已按各自方向对齐）
    # crowding: invert=True（低拥挤度更好）；vol_ratio: invert=False（放量确认更好）；flow: invert=True（资金流出反向更好）
    SIGNAL_DEFS = {"crowding": (crowding, True), "vol_ratio": (vol_ratio, False), "flow": (None, True)}

    signal_ranks = {"crowding": {}, "vol_ratio": {}, "flow": {}}
    for d in rebal_dates:
        crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
        volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)
        flow_d = pd.Series(dtype=float)
        if not flow_1m.empty:
            idx = flow_1m.index[flow_1m.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m.loc[idx[-1]]

        for name, s, invert in [("crowding", crowd_d, True), ("vol_ratio", volr_d, False), ("flow", flow_d, True)]:
            s = s.dropna()
            if len(s) < 5:
                continue
            r = s.rank(pct=True)
            if invert:
                r = 1 - r
            signal_ranks[name][d] = r

    rank_dfs = {name: pd.DataFrame(d).T for name, d in signal_ranks.items()}

    def combo_rank(names):
        parts = [rank_dfs[n] for n in names if not rank_dfs[n].empty]
        if not parts:
            return pd.DataFrame()
        aligned = pd.concat(parts, axis=1, keys=range(len(parts)))
        # 按日期逐行对多个信号取均值（同一标的跨信号平均，缺失的信号跳过）
        combined = {}
        common_dates = parts[0].index
        for other in parts[1:]:
            common_dates = common_dates.union(other.index)
        for d in common_dates:
            vals = [p.loc[d] for p in parts if d in p.index]
            if vals:
                combined[d] = pd.concat(vals, axis=1).mean(axis=1)
        return pd.DataFrame(combined).T

    all_names = ["crowding", "vol_ratio", "flow"]
    subsets = []
    for r in range(1, len(all_names) + 1):
        subsets.extend(itertools.combinations(all_names, r))

    n_days = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]
    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]

    print("\n" + "=" * 90)
    print("信号子集消融实验：动量基线 + 3个候选弱信号的全部非空子集（连续打折叠加）")
    print("=" * 90)

    rows = []
    nav_cache = {}

    nav_base = run_backtest(close, scores, rebal_dates)
    stats_base = calc_stats(nav_base)
    nav_base_is = run_backtest(close_is, sc_is, rebal_is)
    nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos)
    rows.append({"信号子集": "（无，纯动量基线）", "夏普": stats_base["Sharpe"], "年化": stats_base["CAGR"],
                 "回撤": stats_base["MaxDD"], "IS夏普": calc_stats(nav_base_is)["Sharpe"],
                 "OOS夏普": calc_stats(nav_base_oos)["Sharpe"]})
    nav_cache["（无，纯动量基线）"] = nav_base

    for names in subsets:
        label = "+".join(names)
        boost = combo_rank(list(names))
        nav = run_backtest(close, scores, rebal_dates, boost_signal=boost, boost_mode="continuous")
        stats = calc_stats(nav)
        nav_is = run_backtest(close_is, sc_is, rebal_is, boost_signal=boost, boost_mode="continuous")
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, boost_signal=boost, boost_mode="continuous")
        rows.append({"信号子集": label, "夏普": stats["Sharpe"], "年化": stats["CAGR"],
                     "回撤": stats["MaxDD"], "IS夏普": calc_stats(nav_is)["Sharpe"],
                     "OOS夏普": calc_stats(nav_oos)["Sharpe"]})
        nav_cache[label] = nav
        print(f"  完成: {label}  夏普={stats['Sharpe']:.3f}")

    df = pd.DataFrame(rows).set_index("信号子集")
    df_fmt = df.copy()
    df_fmt["夏普"] = df_fmt["夏普"].map(lambda x: f"{x:.3f}")
    df_fmt["年化"] = df_fmt["年化"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["回撤"] = df_fmt["回撤"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["IS夏普"] = df_fmt["IS夏普"].map(lambda x: f"{x:.3f}")
    df_fmt["OOS夏普"] = df_fmt["OOS夏普"].map(lambda x: f"{x:.3f}")
    print("\n" + df_fmt.to_string())

    baseline_sharpe = df.loc["（无，纯动量基线）", "夏普"]
    others = df.drop("（无，纯动量基线）")
    best_label = others["夏普"].idxmax()
    best_sharpe = others.loc[best_label, "夏普"]
    delta = best_sharpe - baseline_sharpe

    print("\n" + "=" * 90)
    print(f"结论：最优信号子集「{best_label}」夏普={best_sharpe:.3f}，基线={baseline_sharpe:.3f}，Δ={delta:+.3f}")
    print("=" * 90)

    # 对最优子集做滚动2年窗口稳健性检验
    if delta > 0.02:
        nav_best = nav_cache[best_label]
        window_days = 252 * 2
        rolling_base, rolling_best = [], []
        for i in range(window_days, len(nav_base)):
            rolling_base.append((nav_base.index[i], roll_sharpe(nav_base.iloc[i - window_days: i])))
        for i in range(window_days, len(nav_best)):
            rolling_best.append((nav_best.index[i], roll_sharpe(nav_best.iloc[i - window_days: i])))
        rs_base = pd.Series(dict(rolling_base))
        rs_best = pd.Series(dict(rolling_best))
        common_idx = rs_base.index.intersection(rs_best.index)
        improvement = rs_best[common_idx] - rs_base[common_idx]
        neg_ratio = (improvement < 0).mean()
        print(f"\n滚动2年夏普稳健性检验（「{best_label}」 vs 基线）：")
        print(f"均值Δ={improvement.mean():+.3f}，std={improvement.std():.3f}，劣于基线占比={neg_ratio:.1%}")
        if neg_ratio > 0.4 or improvement.std() > abs(improvement.mean()) * 2:
            print("→ 不稳健，判定过拟合痕迹，不建议采用。")
        else:
            print("→ 相对稳健，可考虑小规模试用并持续监控。")

    print("\n" + "=" * 90)
    print("全部子集夏普排序：")
    print(others.sort_values("夏普", ascending=False)[["夏普", "年化", "回撤"]].to_string())
    print("=" * 90)


if __name__ == "__main__":
    main()
