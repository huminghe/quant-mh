"""
第十三轮（续2）：对v19消融找出的全局最优子集crowding+flow+margin_balance做
分年度收益/回撤拆解，排除是否被单一年份极端行情主导。

对照组：现有方案crowding+vol_ratio+flow、次优vol_ratio+flow、纯动量基线。
不重复拉取moneyflow_ratio/rate_beta数据（本轮不需要），只拉margin_balance。
"""

import sys
import time
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro

INIT_CASH = 1_000_000
COMMISSION = 0.0001
SLIPPAGE = 0.0002
START_DATE = "2019-01-01"
MOMENTUM_WINDOW = 25
TOP_N = 3
RISK_VOL_WINDOW = 21
CORR_WINDOW = 60
CORR_HIST_WINDOW = 252
SW_INDUSTRY_FILE = pathlib.Path(__file__).parent.parent.parent / "data" / "stock_sw_industry.parquet"

ETF_TO_SECTOR = {
    "515000.SH": "计算机", "512760.SH": "电子", "159995.SZ": "电子",
    "515330.SH": "汽车", "516160.SH": "电力设备", "159629.SZ": "电力设备",
    "159596.SZ": "电力设备", "512010.SH": "医药生物", "512170.SH": "医药生物",
    "159992.SZ": "医药生物", "512800.SH": "银行", "512880.SH": "非银金融",
    "159931.SZ": "房地产", "512980.SH": "传媒", "159869.SZ": "传媒",
    "515030.SH": "电力设备", "159628.SZ": "机械设备", "516670.SH": "有色金属",
    "159975.SZ": "国防军工", "512660.SH": "国防军工", "512400.SH": "有色金属",
    "159928.SZ": "食品饮料", "515700.SH": "食品饮料", "159997.SZ": "食品饮料",
    "159801.SZ": "农林牧渔", "515220.SH": "基础化工", "159611.SZ": "煤炭",
}


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
    daily_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "daily"
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


def get_sw_industry_map() -> dict:
    df = pd.read_parquet(SW_INDUSTRY_FILE)
    return df.set_index("ts_code")["sw_industry"].to_dict()


def fetch_margin_balance_daily(pro, industry_map: dict, trade_dates: list) -> pd.DataFrame:
    rows = {}
    for i, d in enumerate(trade_dates, 1):
        try:
            df = pro.margin_detail(trade_date=d)
            if df.empty:
                continue
            df["sw_industry"] = df["ts_code"].map(industry_map)
            df = df.dropna(subset=["sw_industry"])
            agg = df.groupby("sw_industry")["rzrqye"].sum()
            rows[d] = agg
            time.sleep(0.05)
        except Exception as e:
            print(f"  {d} margin_detail 失败: {e}")
        if i % 200 == 0:
            print(f"  已拉取 {i}/{len(trade_dates)} 天")
    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
                  boost_signal: pd.DataFrame = None, boost_mode: str = "none",
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

        if boost_mode == "continuous" and boost_signal is not None and date in boost_signal.index:
            day_boost = boost_signal.loc[date]
            for code in day_scores.index:
                if code not in day_boost.index or pd.isna(day_boost[code]):
                    continue
                b = day_boost[code]
                day_scores[code] *= (0.5 + b)

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


def yearly_stats(nav: pd.Series) -> pd.DataFrame:
    rows = {}
    for year, group in nav.groupby(nav.index.year):
        if len(group) < 2:
            continue
        ret = group.iloc[-1] / group.iloc[0] - 1
        rolling_max = group.cummax()
        dd = ((group - rolling_max) / rolling_max).min()
        daily_ret = group.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        rows[year] = {"年收益": ret, "年内最大回撤": dd, "年度夏普": sharpe}
    return pd.DataFrame(rows).T


def main():
    print("加载价格与成交额数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]

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
    vol_ratio = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取ETF份额数据（资金流信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    industry_map = get_sw_industry_map()
    trade_dates = [d.strftime("%Y%m%d") for d in close.index]
    print("拉取两融余额数据（margin_balance）...")
    margin_ind = fetch_margin_balance_daily(pro, industry_map, trade_dates)
    margin_ind_monthly = margin_ind.resample("ME").last().pct_change()

    sector_to_etfs = {}
    for etf, sector in ETF_TO_SECTOR.items():
        sector_to_etfs.setdefault(sector, []).append(etf)
    sector_codes = [c for c in valid_codes if c in ETF_TO_SECTOR]

    margin_etf_cols = {}
    for sector, etfs in sector_to_etfs.items():
        if sector not in margin_ind_monthly.columns:
            continue
        for etf in etfs:
            if etf in sector_codes:
                margin_etf_cols[etf] = margin_ind_monthly[sector]
    margin_etf = pd.DataFrame(margin_etf_cols)

    SIGNAL_DEFS = {
        "crowding": (crowding, True), "vol_ratio": (vol_ratio, False),
        "flow": (None, True), "margin_balance": (margin_etf, False),
    }

    signal_ranks = {name: {} for name in SIGNAL_DEFS}
    for d in rebal_dates:
        crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
        volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)
        flow_d = pd.Series(dtype=float)
        if not flow_1m.empty:
            idx = flow_1m.index[flow_1m.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m.loc[idx[-1]]
        margin_d = pd.Series(dtype=float)
        if not margin_etf.empty:
            idx = margin_etf.index[margin_etf.index <= d]
            if len(idx) > 0:
                margin_d = margin_etf.loc[idx[-1]]

        for name, s, invert in [("crowding", crowd_d, True), ("vol_ratio", volr_d, False),
                                 ("flow", flow_d, True), ("margin_balance", margin_d, False)]:
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
        combined = {}
        common_dates = parts[0].index
        for other in parts[1:]:
            common_dates = common_dates.union(other.index)
        for d in common_dates:
            vals = [p.loc[d] for p in parts if d in p.index]
            if vals:
                combined[d] = pd.concat(vals, axis=1).mean(axis=1)
        return pd.DataFrame(combined).T

    print("\n回测4个方案，计算分年度表现...")
    combos_to_check = {
        "纯动量基线": None,
        "crowding+vol_ratio+flow（现有方案）": ["crowding", "vol_ratio", "flow"],
        "vol_ratio+flow": ["vol_ratio", "flow"],
        "crowding+flow+margin_balance（v19最优）": ["crowding", "flow", "margin_balance"],
    }

    yearly_tables = {}
    for label, names in combos_to_check.items():
        if names is None:
            nav = run_backtest(close, scores, rebal_dates)
        else:
            boost = combo_rank(names)
            nav = run_backtest(close, scores, rebal_dates, boost_signal=boost, boost_mode="continuous")
        yearly_tables[label] = yearly_stats(nav)
        print(f"\n【{label}】分年度表现：")
        print(yearly_tables[label].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 90)
    print("汇总：各方案年收益对比（列=方案，行=年份）")
    print("=" * 90)
    ret_compare = pd.DataFrame({label: t["年收益"] for label, t in yearly_tables.items()})
    print(ret_compare.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 90)
    print("v19最优方案 vs 现有方案：逐年收益差")
    print("=" * 90)
    best_label = "crowding+flow+margin_balance（v19最优）"
    base_label = "crowding+vol_ratio+flow（现有方案）"
    diff = ret_compare[best_label] - ret_compare[base_label]
    print(diff.to_string(float_format=lambda x: f"{x:+.3f}"))
    dominant_year = diff.abs().idxmax()
    print(f"\n差距最大年份：{dominant_year}，差距={diff[dominant_year]:+.3f}，"
          f"占全部年份差距绝对值总和的{diff.abs()[dominant_year] / diff.abs().sum():.1%}")


if __name__ == "__main__":
    main()
