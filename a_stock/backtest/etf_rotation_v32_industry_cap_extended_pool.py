"""
行业分散约束在扩容候选池（0.1亿/0.2亿门槛）上的实测

背景：v31冲击成本敏感性验证已确认0.05亿门槛（1312只）优势不稳健（证伪），
0.1亿门槛（1143只）/0.2亿门槛（890只）在中低冲击成本区间全程优于当前1.0亿
门槛（431只），是更稳健的候选池升级方向。v28已在431只候选池上验证cap=1对
分散约束由负贡献反转为正贡献（价格缓存bug修复后），本脚本在两个扩容候选池
上重复同样的验证，行业映射来自 build_etf_sw_exposure_extended.py 新生成的
etf_sw_exposure_01yi.parquet（覆盖1143只，含0.2亿池作为其子集）。

方法：复用 v31 的 build_pool（可变门槛构建候选池+打分），复用 etf_rotation.py
的 use_industry_cap 机制（贪心法，无行业标签标的不受名额限制）。
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_stats, run_backtest, CASH_ETF  # noqa: E402
from etf_rotation_v23_universe_bias_test import TURNOVER_PATH, META_PATH  # noqa: E402
from etf_rotation_v31_impact_cost_sensitivity import build_pool  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_PATH = DATA_DIR / "etf_sw_exposure_01yi.parquet"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

THRESHOLDS_YI = [0.1, 0.2]


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]

    exposure = pd.read_parquet(EXPOSURE_PATH)
    industry_map = exposure.set_index("ts_code")["dominant_industry"].to_dict()

    rows = []
    for th in THRESHOLDS_YI:
        print(f"\n{'=' * 70}\n构建门槛 {th} 亿元候选池...\n{'=' * 70}")
        close, masked_scores, rebal_dates, amount_wide, pool_size = build_pool(turnover, th)
        print(f"候选池规模：{pool_size}只，调仓日数量：{len(rebal_dates)}")

        configs = [
            ("基线（无约束）", dict(use_industry_cap=False)),
            ("行业cap=1", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=1)),
            ("行业cap=2", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=2)),
        ]

        for label, kwargs in configs:
            nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF, **kwargs)
            full_label = f"门槛{th}亿(候选{pool_size}只)_{label}"
            stats = calc_stats(nav, full_label)
            stats["门槛(亿元)"] = th
            stats["候选池规模"] = pool_size
            stats["配置"] = label
            rows.append(stats)
            print(f"  {label}: 夏普={stats['年化夏普']}, 回撤={stats['最大回撤']}")

    df = pd.DataFrame(rows).set_index("标的")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "v32_industry_cap_extended_pool.csv"
    df.to_csv(out_path)

    print("\n" + "=" * 70)
    print("扩容候选池行业分散约束汇总")
    print("=" * 70)
    print(df.to_string())
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
