"""
行业分散约束实测（机械化候选池版）

背景：v27诊断发现机械化候选池（431只）历史上48%的调仓月Top3至少2只同行业，
且这些集中月份贡献了几乎全部策略收益（分散月份简单加总收益接近0）。这只是
事后归因，不能直接回答"加分散约束后夏普会变好还是变差"，因为约束会改变
实际入选的标的（贪心法换掉被挤出的同行业候选，不是简单删除后不补位）。

本脚本在同一个机械化候选池上，用 etf_rotation.py 新增的 use_industry_cap
机制真跑对比：
  基线：无约束（等同 v23 结果，夏普0.59）
  cap=1：同一行业最多1只（等权于 dominant_industry，来自 etf_sw_exposure.parquet）
  cap=2：同一行业最多2只

方法论：全部在431只机械化候选池上验证，不用45只手工池（用户2026-07-27决策）。
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import calc_stats, CASH_ETF  # noqa: E402
from etf_rotation import run_backtest  # noqa: E402
from etf_rotation_v27_concentration_risk import build_universe_close_and_scores  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_PATH = DATA_DIR / "etf_sw_exposure.parquet"


def main():
    print("重建机械化候选池、价格矩阵、动量得分...")
    close, masked_scores, rebal_dates = build_universe_close_and_scores()
    print(f"候选池标的数：{close.shape[1]}，调仓日数量：{len(rebal_dates)}")

    exposure = pd.read_parquet(EXPOSURE_PATH)
    industry_map = exposure.set_index("ts_code")["dominant_industry"].to_dict()

    configs = [
        ("基线（无约束）", dict(use_industry_cap=False)),
        ("行业cap=1", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=1)),
        ("行业cap=2", dict(use_industry_cap=True, industry_map=industry_map, max_per_industry=2)),
    ]

    print("\n" + "=" * 70)
    print("行业分散约束对比（机械化候选池，431只）")
    print("=" * 70)
    rows = []
    for label, kwargs in configs:
        nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF, **kwargs)
        stats = calc_stats(nav, label)
        rows.append(stats)
        print(f"\n{label}:")
        for k, v in stats.items():
            if k != "标的":
                print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(pd.DataFrame(rows).set_index("标的").to_string())


if __name__ == "__main__":
    main()
