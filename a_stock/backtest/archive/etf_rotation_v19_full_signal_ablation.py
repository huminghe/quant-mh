"""
第十三轮（续）：6信号全子集消融实验（把moneyflow_ratio、rate_beta也纳入，
即使它们没通过v17的独立IC初筛）

背景：v18对"crowding+vol_ratio+flow+margin_balance"4信号做了消融，margin_balance
虽IC达标但组合层面无增量贡献。用户追问：moneyflow_ratio和rate_beta在v17里没
通过独立IC筛选就直接被排除了，但第十二轮已有先例——资金流(flow)信号单独用
是负贡献，集成后转正贡献，价值来自"集成"动作本身。因此不能仅凭独立IC筛选
就断言组合层面一定没用，本脚本补做完整6信号全子集消融（63种非空子集）验证。

moneyflow_ratio、rate_beta在v17诊断中IC均值为负（分别-0.0058、-0.0218），
用作boost信号时按方向取反（invert=True），使得排名与预期收益方向一致。

信号�covers范围：
  - crowding/vol_ratio/rate_beta：全部45只ETF
  - flow：全部有份额数据的ETF
  - margin_balance/moneyflow_ratio：仅27只有申万行业映射的行业ETF
"""

import sys
import time
import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd
import akshare as ak

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro

INIT_CASH = 1_000_000
COMMISSION = 0.0001
SLIPPAGE = 0.0002
START_DATE = "2019-01-01"
IS_RATIO = 0.8
MOMENTUM_WINDOW = 25
TOP_N = 3
RISK_VOL_WINDOW = 21
CORR_WINDOW = 60
CORR_HIST_WINDOW = 252
RATE_BETA_WINDOW = 60
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
            print(f"  margin_detail 已拉取 {i}/{len(trade_dates)} 天")
    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def fetch_moneyflow_daily(pro, industry_map: dict, trade_dates: list) -> pd.DataFrame:
    rows = {}
    for i, d in enumerate(trade_dates, 1):
        try:
            df = pro.moneyflow(trade_date=d)
            if df.empty:
                continue
            df["sw_industry"] = df["ts_code"].map(industry_map)
            df = df.dropna(subset=["sw_industry"])
            df["net_lg_amount"] = (df["buy_lg_amount"] + df["buy_elg_amount"]
                                    - df["sell_lg_amount"] - df["sell_elg_amount"])
            df["total_lg_amount"] = (df["buy_lg_amount"] + df["buy_elg_amount"]
                                      + df["sell_lg_amount"] + df["sell_elg_amount"])
            agg = df.groupby("sw_industry").agg(net=("net_lg_amount", "sum"), total=("total_lg_amount", "sum"))
            ratio = agg["net"] / agg["total"].replace(0, np.nan)
            rows[d] = ratio
            time.sleep(0.05)
        except Exception as e:
            print(f"  {d} moneyflow 失败: {e}")
        if i % 200 == 0:
            print(f"  moneyflow 已拉取 {i}/{len(trade_dates)} 天")
    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def load_bond_yield_10y(start_date: str = "20150101") -> pd.Series:
    bond_df = ak.bond_zh_us_rate(start_date=start_date)
    bond_df["日期"] = pd.to_datetime(bond_df["日期"])
    return bond_df.sort_values("日期").set_index("日期")["中国国债收益率10年"]


def calc_rate_beta(close: pd.DataFrame, bond_yield: pd.Series, window: int = RATE_BETA_WINDOW) -> pd.DataFrame:
    ret = close.pct_change()
    yield_diff = bond_yield.diff()
    yield_diff = yield_diff.reindex(ret.index).ffill(limit=5)
    betas = {}
    for code in ret.columns:
        r = ret[code].dropna()
        common = r.index.intersection(yield_diff.dropna().index)
        r = r.loc[common]
        y = yield_diff.loc[common]
        beta_series = pd.Series(index=r.index, dtype=float)
        for i in range(window, len(r)):
            rw = r.iloc[i - window: i].values
            yw = y.iloc[i - window: i].values
            if np.std(yw) < 1e-8:
                continue
            cov = np.cov(rw, yw)[0, 1]
            var = np.var(yw)
            beta_series.iloc[i] = cov / var if var > 1e-12 else np.nan
        betas[code] = beta_series
    return pd.DataFrame(betas).reindex(close.index)


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
    vol_ratio = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取ETF份额数据（资金流信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    industry_map = get_sw_industry_map()
    trade_dates = [d.strftime("%Y%m%d") for d in close.index]

    print("拉取两融余额数据（margin_balance，全市场逐日聚合到行业）...")
    margin_ind = fetch_margin_balance_daily(pro, industry_map, trade_dates)
    margin_ind_monthly = margin_ind.resample("ME").last().pct_change()

    print("拉取大单资金流数据（moneyflow_ratio，全市场逐日聚合到行业）...")
    moneyflow_ind = fetch_moneyflow_daily(pro, industry_map, trade_dates)
    moneyflow_ind_monthly = moneyflow_ind.resample("ME").mean()

    print("加载国债收益率并计算利率敏感度Beta（rate_beta）...")
    bond_yield = load_bond_yield_10y(start_date="20150101")
    rate_beta = calc_rate_beta(close, bond_yield, window=RATE_BETA_WINDOW)

    sector_to_etfs = {}
    for etf, sector in ETF_TO_SECTOR.items():
        sector_to_etfs.setdefault(sector, []).append(etf)
    sector_codes = [c for c in valid_codes if c in ETF_TO_SECTOR]

    def industry_signal_to_etf(ind_signal: pd.DataFrame) -> pd.DataFrame:
        etf_cols = {}
        for sector, etfs in sector_to_etfs.items():
            if sector not in ind_signal.columns:
                continue
            for etf in etfs:
                if etf in sector_codes:
                    etf_cols[etf] = ind_signal[sector]
        return pd.DataFrame(etf_cols)

    margin_etf = industry_signal_to_etf(margin_ind_monthly)
    moneyflow_etf = industry_signal_to_etf(moneyflow_ind_monthly)

    # 信号定义：(数据, invert)。invert=True 表示原始IC为负/反向信号，取反后与"高分看多"方向对齐
    SIGNAL_DEFS = {
        "crowding": (crowding, True),
        "vol_ratio": (vol_ratio, False),
        "flow": (None, True),          # 特殊处理：来自flow_1m，逐日按月对齐
        "margin_balance": (margin_etf, False),
        "moneyflow_ratio": (moneyflow_etf, True),   # v17 IC=-0.0058，取反对齐方向
        "rate_beta": (rate_beta, True),             # v17 IC=-0.0218，取反对齐方向
    }

    print("\n构建月度截面排名信号...")
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
        moneyflow_d = pd.Series(dtype=float)
        if not moneyflow_etf.empty:
            idx = moneyflow_etf.index[moneyflow_etf.index <= d]
            if len(idx) > 0:
                moneyflow_d = moneyflow_etf.loc[idx[-1]]
        rate_beta_d = rate_beta.loc[d] if d in rate_beta.index else pd.Series(dtype=float)

        for name, s, invert in [
            ("crowding", crowd_d, True), ("vol_ratio", volr_d, False), ("flow", flow_d, True),
            ("margin_balance", margin_d, False), ("moneyflow_ratio", moneyflow_d, True),
            ("rate_beta", rate_beta_d, True),
        ]:
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

    all_names = ["crowding", "vol_ratio", "flow", "margin_balance", "moneyflow_ratio", "rate_beta"]
    subsets = []
    for r in range(1, len(all_names) + 1):
        subsets.extend(itertools.combinations(all_names, r))
    print(f"共 {len(subsets)} 种非空子集，开始逐一回测...")

    n_days = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]
    rebal_is = [d for d in rebal_dates if d < split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
    sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]

    print("\n" + "=" * 90)
    print("6信号全子集消融实验：动量基线 + crowding/vol_ratio/flow/margin_balance/moneyflow_ratio/rate_beta")
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

    df = pd.DataFrame(rows).set_index("信号子集")

    baseline_sharpe = df.loc["（无，纯动量基线）", "夏普"]
    others = df.drop("（无，纯动量基线）")
    best_label = others["夏普"].idxmax()
    best_sharpe = others.loc[best_label, "夏普"]
    delta = best_sharpe - baseline_sharpe

    print("\n全部子集夏普排序（前20）：")
    print(others.sort_values("夏普", ascending=False)[["夏普", "年化", "回撤", "IS夏普", "OOS夏普"]]
          .head(20).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 90)
    print(f"结论：63种子集中最优「{best_label}」夏普={best_sharpe:.3f}，基线={baseline_sharpe:.3f}，Δ={delta:+.3f}")
    print("=" * 90)

    for ref_label in ["crowding+vol_ratio+flow", "vol_ratio+flow"]:
        if ref_label in df.index:
            ref_sharpe = df.loc[ref_label, "夏普"]
            print(f"对照：既定方案「{ref_label}」夏普={ref_sharpe:.3f}，新最优Δ={best_sharpe - ref_sharpe:+.3f}")

    # 单独检查含 moneyflow_ratio 或 rate_beta 的子集里表现最好的，判断这两个信号是否有隐藏组合价值
    for extra in ["moneyflow_ratio", "rate_beta"]:
        containing = others[others.index.str.contains(extra)]
        if not containing.empty:
            best_containing = containing["夏普"].idxmax()
            print(f"含「{extra}」的最优子集：「{best_containing}」夏普={containing.loc[best_containing, '夏普']:.3f}")

    if delta > 0.02 and best_label not in ("crowding+vol_ratio+flow", "vol_ratio+flow"):
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
    else:
        print("\n最优子集与既定方案一致或提升幅度<0.02，不做额外滚动窗口检验。")


if __name__ == "__main__":
    main()
