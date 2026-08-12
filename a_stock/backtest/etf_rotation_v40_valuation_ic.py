"""
第十四轮·问题②方向B（估值均值回归）：行业PE/PB分位数信号IC检验（2026-07-27）

背景：方向A（行业景气度基本面，净利润增速/ROE/现金流质量/杠杆/营收增速共
5个指标，见v37/v39）全部排除。用户追问"还有别的基本面指标吗"，估值(PE_TTM/PB)
是完全不同的信号逻辑——不是"景气度在变好"，是"贵不贵"，押注均值回归而非
趋势延续，此前从未在ETF轮动IC框架里测过。

信号构造：全A股月度PE_TTM/PB快照（`fetch_valuation.py`新拉，128个月度调仓日，
非全部日频，只需要截面不需要日内）按申万一级行业groupby中位数聚合，取负值
（估值越低 -> 信号越高 -> 预期跑赢，均值回归方向），通过etf_sw_exposure.parquet
的行业映射复制到ETF。方法/判定标准/ETF映射方案完全复用v37/v39，不重复实现。

与v37/v39的区别：v37/v39用PIT财务数据(merge_asof按公告日匹配)，本脚本用
daily_basic月度截面快照（本身就是point-in-time，不需要额外PIT匹配）。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import get_rebalance_dates, MOMENTUM_WINDOW, START_DATE  # noqa: E402
from etf_rotation_v23_universe_bias_test import load_close_matrix_from_cache  # noqa: E402
from etf_rotation_v25_universe_ensemble_backtest import load_amount_matrix_from_cache  # noqa: E402
from etf_rotation_v37_industry_fundamental_ic import (  # noqa: E402
    EXPOSURE_FILE, SW_INDUSTRY_FILE, MIN_STOCKS_PER_SECTOR,
    get_sw_industry_map, get_etf_industry_map, industry_signal_to_etf,
    evaluate_signal, report_ic, cross_section_corr,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import (  # noqa: E402
    calc_risk_adj_momentum, calc_crowding, fetch_fund_share_all,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
VALUATION_FILE = DATA_DIR / "valuation_monthly.parquet"

VALUATION_FIELDS = {
    "pe_ttm": {"name": "PE_TTM(行业中位数，取负=低估值高分)"},
    "pb":     {"name": "PB(行业中位数，取负=低估值高分)"},
}


def build_industry_valuation_panel(field: str, industry_map: dict) -> pd.DataFrame:
    """月度快照本身就是PIT截面，直接按行业groupby中位数聚合，不需要merge_asof"""
    df = pd.read_parquet(VALUATION_FILE)[["trade_date", "ts_code", field]].dropna()
    df["sw_industry"] = df["ts_code"].map(industry_map)
    df = df.dropna(subset=["sw_industry"])

    grouped = df.groupby(["trade_date", "sw_industry"])[field]
    agg = grouped.median()
    counts = grouped.size()
    agg = agg[counts >= MIN_STOCKS_PER_SECTOR]

    panel = agg.unstack("sw_industry").sort_index()
    return -panel  # 均值回归方向：估值越低信号越高


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
    print(f"其中有申万行业映射的行业ETF：{len(sector_codes)} 只，"
          f"覆盖 {len(set(etf_industry_map[c] for c in sector_codes))} 个申万一级行业")

    fwd_1m = close.pct_change().rolling(21).sum().shift(-21)
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("\n计算风险调整动量（基线，用于冗余检验）...")
    mom_scores = calc_risk_adj_momentum(close_full)[valid_codes]
    mom_scores = mom_scores[mom_scores.index >= START_DATE]

    industry_map = get_sw_industry_map()
    print(f"\n全A股财务数据个股池：{len(industry_map)} 只")

    print(f"\n构建行业估值面板（{len(VALUATION_FIELDS)}个指标）...")
    industry_panels = {}
    for key, cfg in VALUATION_FIELDS.items():
        print(f"  {cfg['name']} ({key})...")
        panel = build_industry_valuation_panel(key, industry_map)
        industry_panels[key] = panel
        print(f"    有效行业数（至少一期非空）：{panel.notna().any().sum()}，评估点数：{len(panel)}")

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
    print("诊断1：行业估值(PE/PB)均值回归信号单独IC（月度截面Rank IC）")
    print("=" * 90)

    results = {}
    for key, panel in industry_panels.items():
        etf_sig = industry_signal_to_etf(panel, etf_industry_map, sector_codes)
        ic = evaluate_signal(etf_sig, fwd_1m, rebal_dates)
        passed, ic_mean = report_ic(VALUATION_FIELDS[key]["name"], ic)
        results[key] = {"signal": etf_sig, "ic": ic, "passed": passed, "ic_mean": ic_mean}

    print("\n" + "=" * 90)
    print("诊断2：与主信号（动量）+ 现有候选信号的截面相关性检验（冗余判定，阈值0.5）")
    print("=" * 90)
    reference_signals = {
        "动量": mom_scores, "crowding": crowding, "vol_ratio": vol_ratio_sig, "flow": flow_sig,
    }
    for key, r in results.items():
        name = VALUATION_FIELDS[key]["name"]
        sig = r["signal"]
        max_abs_corr = 0.0
        max_ref_name = ""
        for ref_name, ref_sig in reference_signals.items():
            if ref_sig.empty:
                continue
            corr_mean = cross_section_corr(sig, ref_sig)
            if pd.isna(corr_mean):
                continue
            print(f"  {name:<30} vs {ref_name:<10}  相关性均值={corr_mean:+.4f}")
            if abs(corr_mean) > abs(max_abs_corr):
                max_abs_corr = corr_mean
                max_ref_name = ref_name
        redundant = abs(max_abs_corr) > 0.5
        print(f"  {name:<30}  最大相关性来自「{max_ref_name}」={max_abs_corr:+.4f}  "
              f"{'冗余（排除）' if redundant else '独立'}")
        results[key]["redundant"] = redundant

    print("\n" + "=" * 90)
    print("诊断3：逐年IC拆解（判断是否存在持续性结构衰减）")
    print("=" * 90)
    for key, r in results.items():
        name = VALUATION_FIELDS[key]["name"]
        ic = r["ic"]
        if ic.empty:
            continue
        yearly = ic.groupby(ic.index.year).agg(["mean", "count"])
        print(f"\n  {name} ({key}):")
        print(yearly.to_string())

    print("\n" + "=" * 90)
    print("最终判定：")
    print("=" * 90)
    survivors = []
    for key, r in results.items():
        name = VALUATION_FIELDS[key]["name"]
        final_pass = r["passed"] and not r["redundant"]
        print(f"  {name:<30}  IC达标={r['passed']}  冗余={r['redundant']}  "
              f"→ {'进入组合消融' if final_pass else '排除'}")
        if final_pass:
            survivors.append(key)

    if survivors:
        print(f"\n通过初筛的信号：{survivors}，可考虑做组合层面消融（参照v18/v38方法论，"
              f"注意分年度拆解看是否有时间结构性衰减）。")
    else:
        print("\n方向B（估值均值回归）均未通过初筛，问题②基本面/估值类信号全部排除。")


if __name__ == "__main__":
    main()
