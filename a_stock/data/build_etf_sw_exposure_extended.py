"""
在 build_etf_sw_exposure.py 基础上，把行业映射覆盖范围从431只（1.0亿门槛候选池）
扩展到0.1亿门槛候选池（1143只），供 etf_rotation_v31 冲击成本验证后的候选池升级方向
（0.1亿/0.2亿门槛）做行业分散约束（cap）测试用。

不修改原 build_etf_sw_exposure.py（其431只候选池文件被v26/v27/v28复用，避免影响既有结论），
候选来源改为 /tmp/pool_01yi_candidates.parquet，输出到独立文件 etf_sw_exposure_01yi.parquet。
"""

import pathlib
import pandas as pd

import build_etf_sw_exposure as base

DATA_DIR = pathlib.Path(__file__).parent
CANDIDATES_PATH = pathlib.Path("/tmp/pool_01yi_candidates.parquet")
OUTPUT_PATH = DATA_DIR / "etf_sw_exposure_01yi.parquet"
OUTPUT_LONG_PATH = DATA_DIR / "etf_sw_exposure_01yi_long.parquet"


def main():
    candidates = pd.read_parquet(CANDIDATES_PATH)["ts_code"].tolist()
    industry_map = base.build_industry_map()

    results = []
    exposure_rows = []
    low_match_warnings = []
    for i, code in enumerate(candidates, 1):
        r = base.classify_etf(code, industry_map)
        results.append({k: v for k, v in r.items() if k != "exposure"})
        for industry, weight in r["exposure"].items():
            exposure_rows.append({"ts_code": code, "sw_industry": industry, "weight": weight,
                                   "end_date": r["end_date"]})
        if r["matched_ratio"] is not None and r["matched_ratio"] < 0.8:
            low_match_warnings.append((code, r["matched_ratio"]))
        if i % 100 == 0:
            print(f"[{i}/{len(candidates)}] 已处理")

    summary = pd.DataFrame(results)
    exposure_long = pd.DataFrame(exposure_rows)

    summary.to_parquet(OUTPUT_PATH, index=False)
    exposure_long.to_parquet(OUTPUT_LONG_PATH, index=False)

    print(f"\n分类汇总（{len(summary)}只）：")
    print(summary["classification"].value_counts())

    print(f"\n匹配率<80%的ETF（{len(low_match_warnings)}只，前20条）：")
    for code, rate in low_match_warnings[:20]:
        print(f"  {code}: {rate:.1%}")

    industry_summary = summary[summary["classification"] == "industry"]
    print(f"\n各申万一级/港股行业代表ETF数量分布（前20）：")
    print(industry_summary["dominant_industry"].value_counts().head(20))

    print(f"\n覆盖到的行业数（含HK:前缀）：{industry_summary['dominant_industry'].nunique()}")
    print(f"\nmissing_cache数量: {(summary['classification'] == 'missing_cache').sum()}")


if __name__ == "__main__":
    main()
