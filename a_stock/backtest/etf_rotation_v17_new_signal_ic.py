"""
第十三轮：3个新候选信号的IC检验

背景：候选池严谨复核（第十二轮，见research.md）确认现有集成信号仅
crowding/vol_ratio/flow三个。本轮补测3个此前未系统检验过的新方向：
  1. 两融余额行业确认（margin_balance）：个股两融余额按申万行业聚合，
     月度环比变化率横截面排名。数据源margin_detail（2015年起）。
  2. 大单资金流行业聚合（moneyflow_ratio）：个股大单+特大单净流入占比
     按申万行业聚合，月度均值横截面排名。数据源moneyflow（全市场逐日）。
  3. 利率敏感度Beta（rate_beta）：ETF日收益对10年期国债收益率变化的
     滚动60日OLS Beta，横截面排名。国债数据源复用v11已验证的akshare
     bond_zh_us_rate接口。

信号①②仅覆盖有申万行业映射的27只行业ETF（复用etf_stock_hybrid_backtest.py
中的ETF_TO_SECTOR映射），宽基/QDII无覆盖，与现有flow/vol_ratio处理方式一致
（信号缺失时该标的不参与boost）。信号③覆盖全部45只ETF。

判定标准（复用v15/项目既定阈值）：
  - |IC均值| >= 0.03 且年度同向占比 >= 60%：通过初筛
  - 与现有3个候选信号（crowding/vol_ratio/flow）或主信号（风险调整动量）
    截面相关性 > 0.5：视为冗余，即使IC达标也排除
  - 不达标：直接排除，不进入组合层面消融（v18）
"""

import sys
import time
import pathlib
import warnings

import numpy as np
import pandas as pd
import akshare as ak

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro
from etf_universe import ETF_CODES

START_DATE = "2019-01-01"
MOMENTUM_WINDOW = 25
RISK_VOL_WINDOW = 21
RATE_BETA_WINDOW = 60
SW_INDUSTRY_FILE = pathlib.Path(__file__).parent.parent / "data" / "stock_sw_industry.parquet"

# 复用 etf_stock_hybrid_backtest.py 中的行业ETF映射
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


def calc_crowding(close: pd.DataFrame, corr_window: int = 60, hist_window: int = 252) -> pd.DataFrame:
    """复用v15/v16已验证实现：ETF间60日滚动相关系数均值 -> 252日历史分位数"""
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
    """复用v15/v16已验证实现：加载ETF日成交额矩阵"""
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
    """复用v15/v16已验证实现：拉取ETF份额数据（资金流信号）"""
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


def cross_section_rank_ic(factor: pd.Series, fwd: pd.Series) -> float:
    common = factor.dropna().index.intersection(fwd.dropna().index)
    if len(common) < 5:
        return np.nan
    return factor[common].corr(fwd[common], method="spearman")


def get_sw_industry_map() -> dict:
    df = pd.read_parquet(SW_INDUSTRY_FILE)
    return df.set_index("ts_code")["sw_industry"].to_dict()


# ── 信号1：两融余额行业确认 ──────────────────────────────────

def fetch_margin_balance_daily(pro, industry_map: dict, trade_dates: list) -> pd.DataFrame:
    """逐日拉取全市场两融余额，按申万一级行业聚合求和，返回 行业名 x 日期 矩阵"""
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
        if i % 100 == 0:
            print(f"  已拉取 {i}/{len(trade_dates)} 天")
    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── 信号2：大单资金流行业聚合 ─────────────────────────────────

def fetch_moneyflow_daily(pro, industry_map: dict, trade_dates: list) -> pd.DataFrame:
    """逐日拉取全市场大单+特大单资金流，按行业聚合净流入占比，返回 行业名 x 日期 矩阵"""
    rows = {}
    for i, d in enumerate(trade_dates, 1):
        try:
            df = pro.moneyflow(trade_date=d)
            if df.empty:
                continue
            df["sw_industry"] = df["ts_code"].map(industry_map)
            df = df.dropna(subset=["sw_industry"])
            df["net_lg_amount"] = (
                df["buy_lg_amount"] + df["buy_elg_amount"]
                - df["sell_lg_amount"] - df["sell_elg_amount"]
            )
            df["total_lg_amount"] = (
                df["buy_lg_amount"] + df["buy_elg_amount"]
                + df["sell_lg_amount"] + df["sell_elg_amount"]
            )
            agg = df.groupby("sw_industry").agg(
                net=("net_lg_amount", "sum"), total=("total_lg_amount", "sum")
            )
            ratio = agg["net"] / agg["total"].replace(0, np.nan)
            rows[d] = ratio
            time.sleep(0.05)
        except Exception as e:
            print(f"  {d} moneyflow 失败: {e}")
        if i % 100 == 0:
            print(f"  已拉取 {i}/{len(trade_dates)} 天")
    result = pd.DataFrame(rows).T
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── 信号3：利率敏感度Beta ────────────────────────────────────

def load_bond_yield_10y(start_date: str = "20150101") -> pd.Series:
    """加载中国10年期国债到期收益率，复用v11 (stock_bond_yield_gap)已验证的数据源"""
    bond_df = ak.bond_zh_us_rate(start_date=start_date)
    bond_df["日期"] = pd.to_datetime(bond_df["日期"])
    return bond_df.sort_values("日期").set_index("日期")["中国国债收益率10年"]


def calc_rate_beta(close: pd.DataFrame, bond_yield: pd.Series, window: int = RATE_BETA_WINDOW) -> pd.DataFrame:
    """滚动窗口OLS：ETF日收益 ~ 国债收益率变化（bp），返回逐日Beta矩阵"""
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


# ── 主流程 ──────────────────────────────────────────────────

def evaluate_signal(name: str, signal: pd.DataFrame, fwd_1m: pd.DataFrame,
                     invert: bool, rebal_dates: list) -> pd.Series:
    """对齐月度截面Rank IC，返回逐月IC序列"""
    ic_list = []
    for d in rebal_dates:
        if d not in fwd_1m.index:
            continue
        idx = signal.index[signal.index <= d]
        if len(idx) == 0:
            continue
        s_d = signal.loc[idx[-1]].dropna()
        if invert:
            s_d = -s_d
        ic = cross_section_rank_ic(s_d, fwd_1m.loc[d])
        if not pd.isna(ic):
            ic_list.append((d, ic))
    return pd.Series(dict(ic_list))


def report_ic(name: str, ic: pd.Series):
    if ic.empty:
        print(f"  {name}: 无有效样本")
        return False, 0.0
    yearly = ic.groupby(ic.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(ic.mean())).mean() if ic.mean() != 0 else 0
    passed = abs(ic.mean()) >= 0.03 and same_sign >= 0.6
    print(f"  {name:<16}  IC均值={ic.mean():+.4f}  IC>0占比={(ic>0).mean():.1%}  "
          f"年度同向占比={same_sign:.1%}  样本={len(ic)}月  "
          f"{'通过初筛' if passed else '未达阈值'}")
    return passed, ic.mean()


def main():
    print("加载ETF价格数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    sector_codes = [c for c in valid_codes if c in ETF_TO_SECTOR]
    print(f"其中有申万行业映射的行业ETF：{len(sector_codes)} 只")

    fwd_1m = close.pct_change().rolling(21).sum().shift(-21)
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("\n计算风险调整动量（基线，用于冗余检验）...")
    mom_scores = calc_risk_adj_momentum(close_full)[valid_codes]
    mom_scores = mom_scores[mom_scores.index >= START_DATE]

    pro = init_pro()
    industry_map = get_sw_industry_map()
    trade_dates = [d.strftime("%Y%m%d") for d in close.index]

    print(f"\n拉取两融余额数据（{len(trade_dates)}个交易日，全市场逐日）...")
    margin_ind = fetch_margin_balance_daily(pro, industry_map, trade_dates)
    margin_ind = margin_ind.pct_change()  # 月度环比在评估时用最近观测值近似（日频数据取月末对月末）
    margin_ind_monthly = margin_ind.resample("ME").last().pct_change()

    print(f"\n拉取大单资金流数据（{len(trade_dates)}个交易日，全市场逐日）...")
    moneyflow_ind = fetch_moneyflow_daily(pro, industry_map, trade_dates)
    moneyflow_ind_monthly = moneyflow_ind.resample("ME").mean()

    print("\n加载国债收益率并计算利率敏感度Beta...")
    bond_yield = load_bond_yield_10y(start_date="20150101")
    rate_beta = calc_rate_beta(close, bond_yield, window=RATE_BETA_WINDOW)

    # 行业信号需要映射回ETF：行业名 -> 该行业对应的ETF代码（可能多对一）
    sector_to_etfs = {}
    for etf, sector in ETF_TO_SECTOR.items():
        sector_to_etfs.setdefault(sector, []).append(etf)

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

    print("\n计算现有3个候选信号（crowding/vol_ratio/flow），用于冗余检验对照...")
    crowding = calc_crowding(close_full[valid_codes])
    crowding = crowding[crowding.index >= START_DATE]
    amount = load_amount_matrix()
    amount = amount[[c for c in valid_codes if c in amount.columns]]
    amount = amount[amount.index >= START_DATE]
    vol_ratio_sig = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_sig = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    print("\n" + "=" * 90)
    print("诊断1：3个新信号单独IC（月度截面Rank IC）")
    print("=" * 90)

    signals_to_test = [
        ("margin_balance", margin_etf, False),
        ("moneyflow_ratio", moneyflow_etf, False),
        ("rate_beta", rate_beta, False),
    ]

    results = {}
    for name, sig, invert in signals_to_test:
        ic = evaluate_signal(name, sig, fwd_1m, invert, rebal_dates)
        passed, ic_mean = report_ic(name, ic)
        results[name] = {"signal": sig, "ic": ic, "passed": passed, "ic_mean": ic_mean, "invert": invert}

    def cross_section_corr(sig_a: pd.DataFrame, sig_b: pd.DataFrame) -> float:
        common_dates = sig_a.index.intersection(sig_b.index)
        corrs = []
        for d in common_dates:
            a = sig_a.loc[d].dropna()
            b = sig_b.loc[d].dropna()
            common_codes = a.index.intersection(b.index)
            if len(common_codes) < 5:
                continue
            corrs.append(a[common_codes].corr(b[common_codes], method="spearman"))
        return np.nanmean(corrs) if corrs else np.nan

    print("\n" + "=" * 90)
    print("诊断2：与主信号（动量）+ 现有3个候选信号的截面相关性检验（冗余判定，阈值0.5）")
    print("=" * 90)
    reference_signals = {
        "动量": mom_scores, "crowding": crowding, "vol_ratio": vol_ratio_sig, "flow": flow_sig,
    }
    for name, r in results.items():
        sig = r["signal"]
        max_abs_corr = 0.0
        max_ref_name = ""
        for ref_name, ref_sig in reference_signals.items():
            if ref_sig.empty:
                continue
            corr_mean = cross_section_corr(sig, ref_sig)
            if pd.isna(corr_mean):
                continue
            print(f"  {name:<16} vs {ref_name:<10}  相关性均值={corr_mean:+.4f}")
            if abs(corr_mean) > abs(max_abs_corr):
                max_abs_corr = corr_mean
                max_ref_name = ref_name
        redundant = abs(max_abs_corr) > 0.5
        print(f"  {name:<16}  最大相关性来自「{max_ref_name}」={max_abs_corr:+.4f}  "
              f"{'冗余（排除）' if redundant else '独立'}")
        results[name]["redundant"] = redundant

    print("\n" + "=" * 90)
    print("最终判定：")
    print("=" * 90)
    survivors = []
    for name, r in results.items():
        final_pass = r["passed"] and not r["redundant"]
        print(f"  {name:<16}  IC达标={r['passed']}  冗余={r['redundant']}  "
              f"→ {'进入组合消融（v18）' if final_pass else '排除'}")
        if final_pass:
            survivors.append(name)

    if survivors:
        print(f"\n通过初筛的信号：{survivors}，请运行 etf_rotation_v18_signal_ablation.py 做组合消融。")
    else:
        print("\n3个新信号均未通过初筛，不进入组合层面消融，判定排除。")


if __name__ == "__main__":
    main()
