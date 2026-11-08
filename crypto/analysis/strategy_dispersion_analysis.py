"""
多策略离散度与整体表现关系：探索性分析（不做触发器，先验证关系方向）

背景：2026-10-21健康度监控调研的后续方向3——"用同策略在不同标的上的离散度
判断是市场问题还是策略特定问题"。调研引用的CTA实证（Kaminski & Yang 2020）
给出一个反直觉结论：trend-following策略间的离散度在表现最好的时期反而
升高（机会多、方向分化大），平淡期才趋同。这与health_monitor.md里"同期
≥2个策略独立触发=系统性问题"的直觉可能相反。

本脚本只做探索性验证：用8个实盘策略（BTC/ETH/SOL/DOGE 各 v2+v3_205m）的
月度PnL（固定资本分母百分比化，可比），计算每月的策略间离散度（标准差），
与当月8策略平均收益的关系做对比，不预设任何阈值，不作为触发器设计的输入。

口径：
  - 每个策略的月度PnL / CAPITAL(20000) * 100，标准化为可比的百分比收益
  - 离散度 = 当月8个策略（有数据的）标准化收益的标准差
  - 按离散度三分组（低/中/高），对比各组当月平均收益、正收益策略占比

用法：
  python strategy_dispersion_analysis.py --dir ~/Downloads/
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from analysis_core import scan_files_auto
from rolling_pl_ratio import read_trade_pnls
from trigger_grid_scan import DEPLOYED_SUFFIXES, monthly_pnl_series

CAPITAL = 20000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/Downloads/")
    args = parser.parse_args()
    directory = args.dir.replace("~", str(Path.home()))

    files = scan_files_auto(directory)
    deployed = {k: v for k, v in files.items() if k.endswith(DEPLOYED_SUFFIXES)}

    # 每个策略的 {年月: 标准化月度收益%}
    per_strategy_monthly: dict[str, dict] = {}
    for key, path in deployed.items():
        trades = read_trade_pnls(path)
        if not trades:
            continue
        monthly = monthly_pnl_series(trades)
        per_strategy_monthly[key] = {ym: pnl / CAPITAL * 100 for ym, pnl in monthly}

    # 汇总所有策略覆盖的年月，要求至少6个策略当月有数据才纳入（早期策略上线时间不同）
    all_months = sorted(set().union(*(m.keys() for m in per_strategy_monthly.values())))
    rows = []
    for ym in all_months:
        vals = [per_strategy_monthly[k][ym] for k in per_strategy_monthly if ym in per_strategy_monthly[k]]
        if len(vals) < 6:
            continue
        dispersion = float(np.std(vals, ddof=1))
        mean_ret = float(np.mean(vals))
        win_share = float(np.mean([v > 0 for v in vals]))
        rows.append({"ym": ym, "dispersion": dispersion, "mean_ret": mean_ret,
                      "win_share": win_share, "n": len(vals)})

    print(f"===== 多策略离散度 vs 整体表现：探索性分析（{len(rows)}个月，"
          f"要求≥6策略覆盖） =====\n")

    dispersions = np.array([r["dispersion"] for r in rows])
    mean_rets = np.array([r["mean_ret"] for r in rows])
    win_shares = np.array([r["win_share"] for r in rows])

    corr = np.corrcoef(dispersions, mean_rets)[0, 1]
    print(f"离散度 vs 当月8策略平均收益：corr={corr:+.3f}\n")

    order = np.argsort(dispersions)
    n = len(order)
    tercile = n // 3
    groups = {
        "低离散度": order[:tercile],
        "中离散度": order[tercile: n - tercile],
        "高离散度": order[n - tercile:],
    }
    print(f"{'分组':<10}{'离散度范围':<20}{'月均收益%':<14}{'正收益策略占比':<16}{'样本月数'}")
    for label, idx in groups.items():
        d_range = f"{dispersions[idx].min():.2f}~{dispersions[idx].max():.2f}"
        print(f"{label:<10}{d_range:<20}{mean_rets[idx].mean():<14.2f}{win_shares[idx].mean()*100:<16.1f}{len(idx)}")

    print(f"\n最高离散度10个月明细（按离散度降序）：")
    top10 = sorted(rows, key=lambda r: -r["dispersion"])[:10]
    for r in top10:
        print(f"  {r['ym']}  离散度={r['dispersion']:.2f}  当月均收益={r['mean_ret']:+.2f}%  "
              f"正收益占比={r['win_share']*100:.0f}%  覆盖{r['n']}策略")

    print(f"\n最低离散度10个月明细（按离散度升序）：")
    bottom10 = sorted(rows, key=lambda r: r["dispersion"])[:10]
    for r in bottom10:
        print(f"  {r['ym']}  离散度={r['dispersion']:.2f}  当月均收益={r['mean_ret']:+.2f}%  "
              f"正收益占比={r['win_share']*100:.0f}%  覆盖{r['n']}策略")


if __name__ == "__main__":
    main()
