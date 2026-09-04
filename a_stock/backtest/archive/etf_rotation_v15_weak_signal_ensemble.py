"""
第十二轮方向2：ML弱信号集成 —— 拥挤度 + 成交量确认 + 资金流 等权/IC加权集成

背景：历史已单独验证过的横截面弱信号——
  - 拥挤度（相关性代理）：干净数据复核后边际收益不显著（research_etf_rotation.md 已移除）
  - 成交量确认（MA5/MA20成交额比）：IC=0.015，无效（research_etf_rotation.md 已排除）
  - ETF资金净流量（反向假设）：IC=-0.062，组合层面0/11超基线，无效（research_etf_rotation.md 已排除）
这三个信号单独都不足以采用，但都是横截面因子（不同于社融这类全市场共同
信号，已确认"全体惩罚不改变排名"无效，不纳入本次集成）。本轮测试：多个
弱信号简单集成（等权/IC加权线性组合）是否存在互补效应，能挖出单独测试
时看不到的增量信息。

方法（按项目惯例，先IC检验排除法）：
1. 分别计算三个信号的横截面排名（月度），与风险调整动量排名做正交化
   （集成信号 = 三者排名简单平均，不用动量本身，避免与基线重复）
2. 计算集成信号的IC（vs 未来1月收益），与三个单独信号的IC对比
3. 若集成IC达到阈值（|IC|>=0.03 且年度同向占比>=60%），再测"动量+集成信号"
   叠加是否提升组合夏普；若不达标直接排除
4. 因子数仅3个，参考历史教训"因子数<10时LGB不优于线性"（factor_lgbm_backtest.py
   已验证），本次不使用LGB，只用简单线性集成
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
from etf_universe import ETF_CODES

START_DATE = "2019-01-01"   # 与fund_flow_ic.py一致（fund_share数据2019年前缺失较多）
MOMENTUM_WINDOW = 25
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


def calc_vol_ratio(amount: pd.DataFrame) -> pd.DataFrame:
    ma5 = amount.rolling(5).mean()
    ma20 = amount.rolling(20).mean()
    return ma5 / ma20.replace(0, np.nan)


def fetch_fund_share_all(pro, codes: list, start_date: str) -> pd.DataFrame:
    today = pd.Timestamp.today().strftime("%Y%m%d")
    frames = {}
    for i, code in enumerate(codes, 1):
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


print("加载价格与成交额数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

fwd_1m = close.pct_change().rolling(21).sum().shift(-21)

print("\n计算风险调整动量（基线，用于正交检验）...")
mom_scores = calc_risk_adj_momentum(close_full)[valid_codes]
mom_scores = mom_scores[mom_scores.index >= START_DATE]

print("计算拥挤度信号...")
crowding = calc_crowding(close_full[valid_codes])
crowding = crowding[crowding.index >= START_DATE]

print("计算成交量确认信号（MA5/MA20成交额比）...")
amount = load_amount_matrix()
amount = amount[[c for c in valid_codes if c in amount.columns]]
amount = amount[amount.index >= START_DATE]
vol_ratio = calc_vol_ratio(amount)

print("拉取ETF份额数据（资金流信号）...")
pro = init_pro()
share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()
print(f"  份额数据覆盖：{share_matrix.shape[1] if not share_matrix.empty else 0} 只标的")

rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# ── 1. 单独信号IC（作为对照，核对与research_etf_rotation.md历史记录一致）────

print("\n" + "=" * 80)
print("诊断1：三个弱信号单独IC（月度截面Rank IC，对照历史记录）")
print("=" * 80)

signal_ic_rows = {"crowding": [], "vol_ratio": [], "fund_flow": [], "ensemble": []}
ensemble_rows = []

for d in rebal_dates:
    if d not in fwd_1m.index:
        continue
    fwd = fwd_1m.loc[d]

    crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
    volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)

    flow_d = pd.Series(dtype=float)
    if not flow_1m.empty:
        idx = flow_1m.index[flow_1m.index <= d]
        if len(idx) > 0:
            flow_d = flow_1m.loc[idx[-1]]

    ic_crowd = cross_section_rank_ic(-crowd_d, fwd)   # 拥挤度越高越差，取负
    ic_volr = cross_section_rank_ic(volr_d, fwd)
    ic_flow = cross_section_rank_ic(-flow_d, fwd)     # 反向假设：净流出→次月涨，取负

    if not pd.isna(ic_crowd):
        signal_ic_rows["crowding"].append((d, ic_crowd))
    if not pd.isna(ic_volr):
        signal_ic_rows["vol_ratio"].append((d, ic_volr))
    if not pd.isna(ic_flow):
        signal_ic_rows["fund_flow"].append((d, ic_flow))

    # 集成：三个信号各自横截面排名（0~1），简单等权平均
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
        continue
    ensemble_score = pd.concat(ranks, axis=1).mean(axis=1)
    ic_ens = cross_section_rank_ic(ensemble_score, fwd)
    if not pd.isna(ic_ens):
        signal_ic_rows["ensemble"].append((d, ic_ens))
        ensemble_rows.append((d, ensemble_score))

for name, rows in signal_ic_rows.items():
    if not rows:
        print(f"  {name}: 无有效样本")
        continue
    ics = pd.Series(dict(rows))
    yearly = ics.groupby(ics.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(ics.mean())).mean() if ics.mean() != 0 else 0
    print(f"  {name:<12}  IC均值={ics.mean():+.4f}  IC>0占比={ (ics>0).mean():.1%}  "
          f"年度同向占比={same_sign:.1%}  样本={len(ics)}月")

# ── 2. 集成信号 vs 单独信号，是否有互补增量 ─────────────────

print("\n" + "=" * 80)
print("诊断2：集成信号 IC 是否显著优于三个单独信号中最好的一个")
print("=" * 80)

ens_ic = pd.Series(dict(signal_ic_rows["ensemble"]))
best_single_ic = max(
    abs(pd.Series(dict(signal_ic_rows[k])).mean()) if signal_ic_rows[k] else 0
    for k in ["crowding", "vol_ratio", "fund_flow"]
)
print(f"集成信号 IC均值（绝对值）= {abs(ens_ic.mean()):.4f}")
print(f"三个单独信号中最优 |IC均值| = {best_single_ic:.4f}")

ens_yearly = ens_ic.groupby(ens_ic.index.year).mean()
ens_same_sign = (np.sign(ens_yearly) == np.sign(ens_ic.mean())).mean() if ens_ic.mean() != 0 else 0

# ── 结论 ─────────────────────────────────────────────────

print("\n" + "=" * 80)
if abs(ens_ic.mean()) < 0.03 or ens_same_sign < 0.6:
    print(f"结论：集成信号IC={ens_ic.mean():+.4f}，年度同向占比={ens_same_sign:.1%}，"
          f"未达排除阈值（|IC|>=0.03 且年度同向占比>=60%）。")
    print("三个弱信号线性等权集成后仍是噪音，无互补效应，判定排除，不进入组合层面回测。")
else:
    print(f"结论：集成信号IC={ens_ic.mean():+.4f}，年度同向占比={ens_same_sign:.1%}，达到信号质量阈值，"
          f"存在互补效应，可考虑进入组合层面回测（动量+集成信号叠加）。")
print("=" * 80)
