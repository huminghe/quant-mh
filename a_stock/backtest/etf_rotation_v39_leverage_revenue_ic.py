"""
第十四轮·问题②方向A续测：杠杆水平(debt_to_assets)+营收增速(or_yoy)行业景气度信号IC检验（2026-07-27）

背景：v37测试的3个财务指标（净利润同比增速/ROE环比变化/现金流质量）均未能
在组合层面（v38消融）产生真实增量贡献，方向A表面证伪。但`financials/`里还有
两个此前从未测试的维度：debt_to_assets（杠杆水平，`fetch_financials.py`早已
下载但没用过）、or_yoy（营业收入同比增速，此前因"资产周转率"缺revenue绝对值
被跳过，但or_yoy只需要同比增速，不需要revenue绝对值，可以补测，已在
`fetch_financials.py`新增字段并重新下载全A股）。

这两个指标覆盖v37未覆盖的维度（杠杆/偿债能力、营收成长），而非v37已测
"盈利能力"类指标（netprofit_yoy/roe_delta/netprofit_margin等）的重复变体，
故值得单独测一轮，而非无差别测完financials/里剩下的全部字段（YAGNI）。

方法/判定标准与ETF映射方案完全复用v37（详见其docstring），本脚本只替换
FUNDAMENTAL_FIELDS，不重复实现行业聚合/IC检验/冗余检验逻辑。
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
    EXPOSURE_FILE, get_sw_industry_map, get_etf_industry_map,
    build_industry_fundamental_panel, industry_signal_to_etf,
    evaluate_signal, report_ic, cross_section_corr,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import (  # noqa: E402
    calc_risk_adj_momentum, calc_crowding, fetch_fund_share_all,
)

NEW_FUNDAMENTAL_FIELDS = {
    "debt_to_assets": {"field": "debt_to_assets", "name": "资产负债率(杠杆水平)", "delta": False},
    "or_yoy":         {"field": "or_yoy",         "name": "营业收入同比增速",     "delta": False},
}


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
    all_stock_codes = list(industry_map.keys())
    print(f"\n全A股财务数据个股池：{len(all_stock_codes)} 只")

    eval_dates = rebal_dates
    print(f"\n构建行业景气度面板（{len(NEW_FUNDAMENTAL_FIELDS)}个指标 x {len(eval_dates)}个月度评估点）...")
    industry_panels = {}
    for key, cfg in NEW_FUNDAMENTAL_FIELDS.items():
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
    print("诊断1：杠杆水平/营收增速信号单独IC（月度截面Rank IC）")
    print("=" * 90)

    results = {}
    for key, panel in industry_panels.items():
        etf_sig = industry_signal_to_etf(panel, etf_industry_map, sector_codes)
        ic = evaluate_signal(etf_sig, fwd_1m, rebal_dates)
        passed, ic_mean = report_ic(NEW_FUNDAMENTAL_FIELDS[key]["name"], ic)
        results[key] = {"signal": etf_sig, "ic": ic, "passed": passed, "ic_mean": ic_mean}

    print("\n" + "=" * 90)
    print("诊断2：与主信号（动量）+ 现有候选信号的截面相关性检验（冗余判定，阈值0.5）")
    print("=" * 90)
    reference_signals = {
        "动量": mom_scores, "crowding": crowding, "vol_ratio": vol_ratio_sig, "flow": flow_sig,
    }
    for key, r in results.items():
        name = NEW_FUNDAMENTAL_FIELDS[key]["name"]
        sig = r["signal"]
        max_abs_corr = 0.0
        max_ref_name = ""
        for ref_name, ref_sig in reference_signals.items():
            if ref_sig.empty:
                continue
            corr_mean = cross_section_corr(sig, ref_sig)
            if pd.isna(corr_mean):
                continue
            print(f"  {name:<20} vs {ref_name:<10}  相关性均值={corr_mean:+.4f}")
            if abs(corr_mean) > abs(max_abs_corr):
                max_abs_corr = corr_mean
                max_ref_name = ref_name
        redundant = abs(max_abs_corr) > 0.5
        print(f"  {name:<20}  最大相关性来自「{max_ref_name}」={max_abs_corr:+.4f}  "
              f"{'冗余（排除）' if redundant else '独立'}")
        results[key]["redundant"] = redundant

    print("\n" + "=" * 90)
    print("诊断3：逐年IC拆解（判断是否存在v17式持续性结构衰减）")
    print("=" * 90)
    for key, r in results.items():
        name = NEW_FUNDAMENTAL_FIELDS[key]["name"]
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
        name = NEW_FUNDAMENTAL_FIELDS[key]["name"]
        final_pass = r["passed"] and not r["redundant"]
        print(f"  {name:<20}  IC达标={r['passed']}  冗余={r['redundant']}  "
              f"→ {'进入组合消融' if final_pass else '排除'}")
        if final_pass:
            survivors.append(key)

    if survivors:
        print(f"\n通过初筛的信号：{survivors}，可考虑做组合层面消融（参照v18/v38方法论，"
              f"注意分年度拆解看是否有时间结构性衰减）。")
    else:
        print("\n杠杆水平/营收增速两个指标均未通过初筛，方向A剩余候选指标排除。")


if __name__ == "__main__":
    main()
