"""
第十四轮·问题②方向A：行业景气度基本面聚合信号IC检验（2026-07-27）

背景：用户明确拍板"肯定不能用45只池子"后提出三个悬而未决问题，本脚本处理
第二个——"是否也可以有别的策略来选取，不局限于动量策略"。外部调研后汇总
3个候选子方向（行业景气度基本面/分析师预期行业聚合/ETF份额申购赎回），
用户选定本方向优先测试。

方法：把个股财务指标（净利润同比增速/ROE环比变化/现金流质量）用PIT方式
取值，按申万一级行业groupby聚合（中位数，抗异常值），再通过
etf_sw_exposure.parquet的行业映射复制到该行业下的ETF，做月度截面Rank IC检验。

与历史已测方向的区别（避免重复劳动）：
  - 与个股层面因子研究（factor_ic_quality_v2等）不同：那是沪深300/中证500
    个股选股（指数增强场景），本方向是ETF行业轮动场景下的行业级聚合信号。
  - 与v17两融余额/资金流行业聚合不同：那是资金面信号，本方向是财务基本面。
  - 与v26申万行业分层候选池不同：那是"候选池构建规则"（砍到每行业1-2只），
    本方向是"候选池内的选股信号"，用途完全不同。

数据前提：financials/已从1574只（沪深300+中证500）扩容到全A股5864只
（2026-07-27新下载，覆盖率从最低11.5%提升到全部>=92%），解决了机械设备/
轻工制造等行业此前只覆盖十几只大盘股、聚合信号无代表性的问题。

指标清单（三个，均可从现有financials/字段直接取，不新拉数据）：
  - netprofit_yoy：净利润同比增速（成长）
  - roe_delta：ROE_TTM(roe_dt)环比变化（边际改善方向，非绝对水平）
  - ocf_to_profit：经营现金流/净利润（盈利质量，个股层面已验证ICIR+0.219）
  资产周转率因financials/和balancesheet/均无营业收入(revenue)字段，跳过。

ETF→行业映射：用etf_sw_exposure.parquet（218只industry类，覆盖24个申万
一级行业），比v17硬编码的27只ETF_TO_SECTOR覆盖更广。

判定标准（复用v17/项目既定阈值）：
  - |IC均值| >= 0.03 且年度同向占比 >= 60%：通过初筛
  - 与现有信号（动量/crowding/vol_ratio/flow）截面相关性 > 0.5：视为冗余
  - 不达标：直接排除，不进入组合层面消融
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402
from fetch_financials import load_financials  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import calc_all_scores, get_rebalance_dates, MOMENTUM_WINDOW, START_DATE  # noqa: E402
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH, CACHE_DIR, load_close_matrix_from_cache,
)
from etf_rotation_v25_universe_ensemble_backtest import load_amount_matrix_from_cache  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import (  # noqa: E402
    calc_risk_adj_momentum, calc_crowding, cross_section_rank_ic,
    fetch_fund_share_all,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
SW_INDUSTRY_FILE = DATA_DIR / "stock_sw_industry.parquet"
EXPOSURE_FILE = DATA_DIR / "etf_all_candidates.parquet"
EXPOSURE_PATH = DATA_DIR / "etf_sw_exposure.parquet"

MIN_STOCKS_PER_SECTOR = 5  # 行业内最少有效个股数，低于则该行业当期不聚合（填NaN）

FUNDAMENTAL_FIELDS = {
    "netprofit_yoy": {"field": "netprofit_yoy", "name": "净利润同比增速", "delta": False},
    "roe_delta":      {"field": "roe_dt",        "name": "ROE环比变化",     "delta": True},
    "ocf_to_profit":  {"field": "ocf_to_profit",  "name": "现金流质量(OCF/NI)", "delta": False},
}


# ── 1. 行业映射 ──────────────────────────────────────────────

def get_sw_industry_map() -> dict:
    df = pd.read_parquet(SW_INDUSTRY_FILE)
    return df.set_index("ts_code")["sw_industry"].to_dict()


def get_etf_industry_map() -> dict:
    """ETF -> 申万一级行业（用etf_sw_exposure.parquet的industry类，排除港股HK:前缀）"""
    df = pd.read_parquet(EXPOSURE_PATH)
    ind = df[(df["classification"] == "industry") & (~df["dominant_industry"].str.startswith("HK:"))]
    return ind.set_index("ts_code")["dominant_industry"].to_dict()


# ── 2. 行业景气度面板（PIT，按季度报告期聚合后ffill到月度评估点）──

def build_industry_fundamental_panel(codes: list, industry_map: dict,
                                      field: str, use_delta: bool,
                                      eval_dates: list) -> pd.DataFrame:
    """
    对每个评估日期，取每只个股PIT最新值（use_delta时取相对上一期的差值），
    按行业groupby中位数聚合。返回 index=eval_date，columns=行业名。

    用 pd.merge_asof(by="ts_code") 做向量化PIT匹配，避免对~5800只个股 x
    ~120个评估点做逐个股逐日期的Python嵌套循环（原实现在全A股规模下过慢，
    单指标测算超过30分钟未完成，改用merge_asof后是C层面的向量化操作）。
    """
    long_frames = []
    for code in codes:
        df = load_financials(code)
        if df.empty or field not in df.columns:
            continue
        sub = df[["ann_date", field]].dropna().sort_values("ann_date")
        if sub.empty:
            continue
        if use_delta:
            sub = sub.copy()
            sub["value"] = sub[field].diff()
            sub = sub.dropna(subset=["value"])
        else:
            sub = sub.rename(columns={field: "value"})
        if sub.empty:
            continue
        sub = sub[["ann_date", "value"]].copy()
        sub["ts_code"] = code
        long_frames.append(sub)

    if not long_frames:
        return pd.DataFrame(index=eval_dates)

    financial_long = pd.concat(long_frames, ignore_index=True).sort_values("ann_date")

    # 左表：codes x eval_dates 的全组合，按ts_code分组内按eval_date升序（merge_asof by=要求）
    codes_with_data = financial_long["ts_code"].unique()
    # merge_asof 要求 left/right 均按 "on" 列（eval_date/ann_date）整体升序，
    # by=ts_code 只是在匹配时按组过滤，不代表可以只按ts_code排序
    left = pd.DataFrame({
        "ts_code": np.repeat(codes_with_data, len(eval_dates)),
        "eval_date": np.tile(pd.to_datetime(eval_dates), len(codes_with_data)),
    }).sort_values(["eval_date", "ts_code"]).reset_index(drop=True)

    financial_long = financial_long.sort_values(["ann_date", "ts_code"]).reset_index(drop=True)

    matched = pd.merge_asof(
        left, financial_long,
        left_on="eval_date", right_on="ann_date",
        by="ts_code", direction="backward",
    )
    matched = matched.dropna(subset=["value"])
    matched["sw_industry"] = matched["ts_code"].map(industry_map)
    matched = matched.dropna(subset=["sw_industry"])

    grouped = matched.groupby(["eval_date", "sw_industry"])["value"]
    agg = grouped.median()
    counts = grouped.size()
    agg = agg[counts >= MIN_STOCKS_PER_SECTOR]

    panel = agg.unstack("sw_industry").sort_index()
    return panel


def industry_signal_to_etf(ind_signal: pd.DataFrame, etf_industry_map: dict,
                           valid_etfs: list) -> pd.DataFrame:
    """行业名 x 日期 -> ETF代码 x 日期（同行业内所有ETF共享该行业信号）"""
    etf_cols = {}
    for etf in valid_etfs:
        sector = etf_industry_map.get(etf)
        if sector is not None and sector in ind_signal.columns:
            etf_cols[etf] = ind_signal[sector]
    return pd.DataFrame(etf_cols)


# ── 3. IC检验 + 冗余检验（复用v17判定标准）───────────────────

def evaluate_signal(signal: pd.DataFrame, fwd_1m: pd.DataFrame, rebal_dates: list) -> pd.Series:
    ic_list = []
    for d in rebal_dates:
        if d not in fwd_1m.index:
            continue
        idx = signal.index[signal.index <= d]
        if len(idx) == 0:
            continue
        s_d = signal.loc[idx[-1]].dropna()
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


def main():
    print("加载ETF候选池与价格缓存...")
    all_candidates = pd.read_parquet(EXPOSURE_FILE)["ts_code"].tolist()
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    etf_industry_map = get_etf_industry_map()
    sector_codes = [c for c in valid_codes if c in etf_industry_map]
    print(f"其中有申万行业映射（etf_sw_exposure.parquet）的行业ETF：{len(sector_codes)} 只，"
          f"覆盖 {len(set(etf_industry_map[c] for c in sector_codes))} 个申万一级行业")

    fwd_1m = close.pct_change().rolling(21).sum().shift(-21)
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("\n计算风险调整动量（基线，用于冗余检验）...")
    mom_scores = calc_risk_adj_momentum(close_full)[valid_codes]
    mom_scores = mom_scores[mom_scores.index >= START_DATE]

    industry_map = get_sw_industry_map()
    all_stock_codes = list(industry_map.keys())
    print(f"\n全A股财务数据个股池：{len(all_stock_codes)} 只")

    eval_dates = rebal_dates
    print(f"\n构建行业景气度面板（{len(FUNDAMENTAL_FIELDS)}个指标 x {len(eval_dates)}个月度评估点）...")
    industry_panels = {}
    for key, cfg in FUNDAMENTAL_FIELDS.items():
        print(f"  {cfg['name']} ({key})...")
        panel = build_industry_fundamental_panel(
            all_stock_codes, industry_map, cfg["field"], cfg["delta"], eval_dates
        )
        industry_panels[key] = panel
        print(f"    有效行业数（至少一期非空）：{panel.notna().any().sum()}")

    print("\n计算现有候选信号（crowding/vol_ratio/flow），用于冗余检验对照...")
    crowding = calc_crowding(close_full[valid_codes])
    crowding = crowding[crowding.index >= START_DATE]
    amount = load_amount_matrix_from_cache(valid_codes)
    amount = amount[amount.index >= START_DATE]
    vol_ratio_sig = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_sig = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    print("\n" + "=" * 90)
    print("诊断1：行业景气度基本面信号单独IC（月度截面Rank IC）")
    print("=" * 90)

    results = {}
    for key, panel in industry_panels.items():
        etf_sig = industry_signal_to_etf(panel, etf_industry_map, sector_codes)
        ic = evaluate_signal(etf_sig, fwd_1m, rebal_dates)
        passed, ic_mean = report_ic(FUNDAMENTAL_FIELDS[key]["name"], ic)
        results[key] = {"signal": etf_sig, "ic": ic, "passed": passed, "ic_mean": ic_mean}

    print("\n" + "=" * 90)
    print("诊断2：与主信号（动量）+ 现有候选信号的截面相关性检验（冗余判定，阈值0.5）")
    print("=" * 90)
    reference_signals = {
        "动量": mom_scores, "crowding": crowding, "vol_ratio": vol_ratio_sig, "flow": flow_sig,
    }
    for key, r in results.items():
        name = FUNDAMENTAL_FIELDS[key]["name"]
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
        results[key]["redundant"] = redundant

    print("\n" + "=" * 90)
    print("最终判定：")
    print("=" * 90)
    survivors = []
    for key, r in results.items():
        name = FUNDAMENTAL_FIELDS[key]["name"]
        final_pass = r["passed"] and not r["redundant"]
        print(f"  {name:<16}  IC达标={r['passed']}  冗余={r['redundant']}  "
              f"→ {'进入组合消融' if final_pass else '排除'}")
        if final_pass:
            survivors.append(key)

    if survivors:
        print(f"\n通过初筛的信号：{survivors}，可考虑做组合层面消融（参照v18方法论，"
              f"注意分年度拆解看是否有时间结构性衰减）。")
    else:
        print("\n方向A（行业景气度基本面）全部指标均未通过初筛，判定排除。")


if __name__ == "__main__":
    main()
