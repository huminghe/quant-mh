"""
计算每只候选ETF的行业暴露分布与集中度，用于替代人工挑选行业的分层候选池方案。

数据来源：
- fund_portfolio_cache/{ts_code}.parquet（fetch_etf_portfolio_holdings.py 拉取的季度持仓明细）
- stock_sw_industry.parquet（A股个股→申万一级行业映射，31个行业）
- hk_industry.parquet（港股→同花顺行业指数映射，67个行业，fetch_hk_industry.py 拉取）

关键处理：
- 中国基金季报（Q1/Q3，end_date以0331/0930结尾）只披露前十大持仓（stk_mkv_ratio加总约50-58%），
  半年报/年报（Q2/Q4，end_date以0630/1231结尾）才披露完整持仓（加总接近100%）。
  用不完整的季报算集中度会系统性失真，因此优先选最近一期"完整披露"（加总>=90%）的持仓快照。
  若某ETF历史上从无完整披露（只有季报），退化使用最新一期季报并标注 disclosure_complete=False。
- fund_portfolio为空（QDII/跨境ETF、新发行等）直接归为"broad_qdii"。
- 持仓按代码后缀分流匹配：A股（.SZ/.SH）匹配申万一级，港股（.HK）匹配同花顺行业指数，
  两套体系行业名不同源，用"HK:"前缀区分（如"HK:半导体产品与设备"），不强行合并到申万一级。
  仍未匹配到任何行业的持仓（可转债/现金等）计入"未分类"，不进入任何行业分子，
  但计入 matched_ratio 分母以便诊断匹配率。

分类规则：concentration >= THRESHOLD 判定为"行业代表"（classification="industry"，
归入 dominant_industry），否则归入"broad"（宽基/多元化）。
"""

import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent
CANDIDATES_PATH = DATA_DIR / "etf_all_candidates.parquet"
HOLDINGS_DIR = DATA_DIR / "fund_portfolio_cache"
SW_INDUSTRY_PATH = DATA_DIR / "stock_sw_industry.parquet"
HK_INDUSTRY_PATH = DATA_DIR / "hk_industry.parquet"
OUTPUT_PATH = DATA_DIR / "etf_sw_exposure.parquet"

DISCLOSURE_COMPLETE_THRESHOLD = 90.0  # stk_mkv_ratio加总>=90%才算"完整披露"
CONCENTRATION_THRESHOLD = 0.35  # 初始阈值，跑完后看分布直方图+常识校验再调整


def pick_latest_snapshot(df: pd.DataFrame) -> tuple:
    """从一只ETF的全历史持仓里选出用于分类的那一期快照。
    优先选最近一期"完整披露"（加总>=90%）的end_date；若没有则退化为最新一期，并标记不完整。
    """
    period_sum = df.groupby("end_date")["stk_mkv_ratio"].sum()
    complete_periods = period_sum[period_sum >= DISCLOSURE_COMPLETE_THRESHOLD]
    if not complete_periods.empty:
        end_date = complete_periods.index.max()
        return end_date, True
    end_date = period_sum.index.max()
    return end_date, False


def classify_etf(ts_code: str, industry_map: pd.DataFrame) -> dict:
    """industry_map: 列 [symbol, industry]，A股+港股合并后的统一持仓→行业映射（港股行业名带"HK:"前缀）。"""
    path = HOLDINGS_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return {"ts_code": ts_code, "classification": "missing_cache", "end_date": None,
                "disclosure_complete": None, "concentration": None, "dominant_industry": None,
                "matched_ratio": None, "exposure": {}}

    df = pd.read_parquet(path)
    if df.empty:
        return {"ts_code": ts_code, "classification": "broad_qdii", "end_date": None,
                "disclosure_complete": None, "concentration": None, "dominant_industry": None,
                "matched_ratio": None, "exposure": {}}

    end_date, disclosure_complete = pick_latest_snapshot(df)
    snap = df[df["end_date"] == end_date].copy()
    snap = snap.merge(industry_map, left_on="symbol", right_on="symbol", how="left")

    total_ratio = snap["stk_mkv_ratio"].sum()
    matched_ratio = snap.loc[snap["industry"].notna(), "stk_mkv_ratio"].sum()
    match_rate = matched_ratio / total_ratio if total_ratio > 0 else 0.0

    exposure = snap.groupby("industry")["stk_mkv_ratio"].sum() / 100.0
    exposure = exposure.sort_values(ascending=False)

    if exposure.empty:
        return {"ts_code": ts_code, "classification": "broad", "end_date": end_date,
                "disclosure_complete": disclosure_complete, "concentration": 0.0,
                "dominant_industry": None, "matched_ratio": match_rate, "exposure": {}}

    dominant_industry = exposure.idxmax()
    concentration = exposure.max()
    classification = "industry" if concentration >= CONCENTRATION_THRESHOLD else "broad"

    return {"ts_code": ts_code, "classification": classification, "end_date": end_date,
            "disclosure_complete": disclosure_complete, "concentration": concentration,
            "dominant_industry": dominant_industry, "matched_ratio": match_rate,
            "exposure": exposure.to_dict()}


def build_industry_map() -> pd.DataFrame:
    """合并A股（申万一级）+ 港股（同花顺行业指数，加"HK:"前缀区分体系）持仓映射，统一列名为 symbol/industry。"""
    sw_map = pd.read_parquet(SW_INDUSTRY_PATH).rename(
        columns={"ts_code": "symbol", "sw_industry": "industry"})
    hk_map = pd.read_parquet(HK_INDUSTRY_PATH).rename(
        columns={"ts_code": "symbol", "hk_industry": "industry"})
    hk_map["industry"] = "HK:" + hk_map["industry"]
    return pd.concat([sw_map[["symbol", "industry"]], hk_map[["symbol", "industry"]]], ignore_index=True)


def main():
    candidates = pd.read_parquet(CANDIDATES_PATH)["ts_code"].tolist()
    industry_map = build_industry_map()

    results = []
    exposure_rows = []
    low_match_warnings = []
    for i, code in enumerate(candidates, 1):
        r = classify_etf(code, industry_map)
        results.append({k: v for k, v in r.items() if k != "exposure"})
        for industry, weight in r["exposure"].items():
            exposure_rows.append({"ts_code": code, "sw_industry": industry, "weight": weight,
                                   "end_date": r["end_date"]})
        if r["matched_ratio"] is not None and r["matched_ratio"] < 0.8:
            low_match_warnings.append((code, r["matched_ratio"]))
        if i % 50 == 0:
            print(f"[{i}/{len(candidates)}] 已处理")

    summary = pd.DataFrame(results)
    exposure_long = pd.DataFrame(exposure_rows)

    summary.to_parquet(OUTPUT_PATH, index=False)
    exposure_long.to_parquet(DATA_DIR / "etf_sw_exposure_long.parquet", index=False)

    print(f"\n分类汇总（{len(summary)}只）：")
    print(summary["classification"].value_counts())

    print(f"\n匹配率<80%的ETF（{len(low_match_warnings)}只，前20条）：")
    for code, rate in low_match_warnings[:20]:
        print(f"  {code}: {rate:.1%}")

    industry_summary = summary[summary["classification"] == "industry"]
    print(f"\n各申万一级行业代表ETF数量分布：")
    print(industry_summary["dominant_industry"].value_counts())

    print(f"\n覆盖到的申万一级行业数：{industry_summary['dominant_industry'].nunique()} / 31")

    # 常识校验：已知宽基/行业ETF的分类结果
    sanity_check_codes = {
        "510300.SH": "沪深300(应为broad)", "510500.SH": "中证500(应为broad)",
        "510050.SH": "上证50(应为broad)", "512760.SH": "半导体ETF(应为industry/电子)",
        "159997.SZ": "白酒ETF(应为industry/食品饮料)", "512660.SH": "军工ETF(应为industry/国防军工)",
    }
    print("\n常识校验：")
    for code, desc in sanity_check_codes.items():
        row = summary[summary["ts_code"] == code]
        if row.empty:
            print(f"  {code} {desc}: 不在候选池中")
            continue
        row = row.iloc[0]
        print(f"  {code} {desc}: classification={row['classification']}, "
              f"concentration={row['concentration']}, dominant={row['dominant_industry']}, "
              f"disclosure_complete={row['disclosure_complete']}")


if __name__ == "__main__":
    main()
