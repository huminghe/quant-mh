"""
趋势强度连续仓位缩放：历史回算

背景：2026-10-21健康度监控调研证伪了"策略PnL统计量触发二元暂停"这条路线。
后续方向验证（trend_persistence_signal.py）发现 ADX(14) 和效率比 ER 在全部
8个实盘策略上都与当月PnL正相关（corr全为正，SOL_v3_205m/BTC_v2最强约
+0.5），Hurst exponent无一致性已排除。

本脚本做第二步：把这个描述性关系转成"连续仓位折扣"方案并回算，验证是否
真的能提升风险调整后收益——而不是只停留在"指标和PnL相关"这个观察。

折扣映射（线性，不是二元开关）：
  factor = clip((adx - LOW) / (HIGH - LOW), FLOOR, 1.0)
  ADX<LOW时折扣到FLOOR（不是0，避免错过趋势刚启动的早期阶段）
  ADX>HIGH时满仓（1.0）

注意：这与已上线的ATR定仓（方法A，按波动率水平反向调整仓位，解决"同一
regime里每笔风险多大"）是不同维度的信号，不是重复验证同一件事——这里用
的是趋势强度（机制判断），不是波动率水平（风险平衡）。两者理论上可叠加，
但本脚本只验证趋势强度这一层，不叠加ATR，避免混淆两个信号的贡献。

口径：
  - 用上个月月末的ADX/ER值决定下个月的仓位折扣（避免用未来数据，只能用
    已确定的上月末状态），当月交易的pnl统一乘以该折扣系数
  - 对比：折扣版 vs 原始版（无折扣，factor恒为1）的累计收益、
    月度Sharpe（均值/标准差，不年化，口径对齐health_monitor.md）、最大回撤

用法：
  python trend_scaling_replay.py --dir ~/Downloads/
  python trend_scaling_replay.py --dir ~/Downloads/ --signal er
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
from trigger_grid_scan import DEPLOYED_SUFFIXES
from trend_persistence_signal import ASSETS, SYMBOL_MAP, fetch_daily_ohlcv, monthly_signal_series

# 折扣映射参数：ADX口径（标准技术指标常用阈值20/25）与ER口径（0-1原生范围）不同
SCALING_PARAMS = {
    "adx": {"low": 15.0, "high": 30.0, "floor": 0.4},
    "er": {"low": 0.15, "high": 0.40, "floor": 0.4},
}


def scaling_factor(value: float, low: float, high: float, floor: float) -> float:
    if value is None:
        return 1.0  # 信号缺失时不缩放，保守默认满仓
    raw = (value - low) / (high - low)
    return float(np.clip(raw, floor, 1.0))


def max_drawdown_from_monthly(monthly_pnls: list[float], capital: float) -> float:
    """用月度净PnL累计序列算最大回撤（固定资本分母百分比）。"""
    cum = np.cumsum(monthly_pnls) / capital * 100
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return dd.min() if len(dd) else 0.0


def sharpe_from_monthly(monthly_pnls: list[float]) -> float | None:
    if len(monthly_pnls) < 2:
        return None
    std = np.std(monthly_pnls, ddof=1)
    return float(np.mean(monthly_pnls) / std) if std > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/Downloads/")
    parser.add_argument("--signal", choices=["adx", "er"], default="adx")
    args = parser.parse_args()
    directory = args.dir.replace("~", str(Path.home()))
    params = SCALING_PARAMS[args.signal]

    files = scan_files_auto(directory)
    deployed = {k: v for k, v in files.items() if k.endswith(DEPLOYED_SUFFIXES)}

    print(f"===== 趋势强度连续仓位缩放回算（信号={args.signal}，"
          f"low={params['low']} high={params['high']} floor={params['floor']}） =====\n")

    CAPITAL = 20000.0
    summary_rows = []

    for asset in ASSETS:
        symbol = SYMBOL_MAP[asset]
        df = fetch_daily_ohlcv(symbol)
        signal_series = monthly_signal_series(df)
        signal_map = {ym: v for ym, v in signal_series}

        strat_keys = sorted(k for k in deployed if k.startswith(asset + "_"))
        for key in strat_keys:
            trades = read_trade_pnls(deployed[key])
            if not trades:
                continue

            # 按月分组交易，月度折扣系数用"上个月末"的信号值（避免用未来数据）
            months_sorted = sorted({(d.year, d.month) for d, _ in trades})
            month_order = {ym: i for i, ym in enumerate(months_sorted)}

            orig_monthly = {}
            scaled_monthly = {}
            for d, p in trades:
                ym = (d.year, d.month)
                idx = month_order[ym]
                orig_monthly.setdefault(ym, 0.0)
                scaled_monthly.setdefault(ym, 0.0)
                orig_monthly[ym] += p

                if idx == 0:
                    factor = 1.0  # 第一个月无上月信号可用，不缩放
                else:
                    prev_ym = months_sorted[idx - 1]
                    sig = signal_map.get(prev_ym)
                    value = sig[args.signal] if sig else None
                    factor = scaling_factor(value, params["low"], params["high"], params["floor"])
                scaled_monthly[ym] += p * factor

            if len(orig_monthly) < 12:
                continue

            orig_vals = [orig_monthly[ym] for ym in months_sorted]
            scaled_vals = [scaled_monthly[ym] for ym in months_sorted]

            orig_total = sum(orig_vals) / CAPITAL * 100
            scaled_total = sum(scaled_vals) / CAPITAL * 100
            orig_sharpe = sharpe_from_monthly(orig_vals)
            scaled_sharpe = sharpe_from_monthly(scaled_vals)
            orig_mdd = max_drawdown_from_monthly(orig_vals, CAPITAL)
            scaled_mdd = max_drawdown_from_monthly(scaled_vals, CAPITAL)

            print(f"--- {key}（{len(months_sorted)}个月，{months_sorted[0]}~{months_sorted[-1]}） ---")
            print(f"    原始：  累计收益={orig_total:+.1f}%  Sharpe={orig_sharpe if orig_sharpe is None else f'{orig_sharpe:.3f}'}  最大回撤={orig_mdd:.1f}%")
            print(f"    缩放后：累计收益={scaled_total:+.1f}%  Sharpe={scaled_sharpe if scaled_sharpe is None else f'{scaled_sharpe:.3f}'}  最大回撤={scaled_mdd:.1f}%")
            sharpe_diff = (scaled_sharpe - orig_sharpe) if (orig_sharpe is not None and scaled_sharpe is not None) else None
            print(f"    差值：  收益{scaled_total - orig_total:+.1f}pp  Sharpe{'N/A' if sharpe_diff is None else f'{sharpe_diff:+.3f}'}  回撤改善{scaled_mdd - orig_mdd:+.1f}pp\n")

            summary_rows.append({
                "key": key, "orig_total": orig_total, "scaled_total": scaled_total,
                "orig_sharpe": orig_sharpe, "scaled_sharpe": scaled_sharpe,
                "orig_mdd": orig_mdd, "scaled_mdd": scaled_mdd,
            })

    print("===== 汇总 =====")
    sharpe_improved = sum(1 for r in summary_rows if r["orig_sharpe"] and r["scaled_sharpe"] and r["scaled_sharpe"] > r["orig_sharpe"])
    mdd_improved = sum(1 for r in summary_rows if r["scaled_mdd"] > r["orig_mdd"])
    total_improved = sum(1 for r in summary_rows if r["scaled_total"] > r["orig_total"])
    n = len(summary_rows)
    print(f"策略数={n}  Sharpe改善={sharpe_improved}/{n}  最大回撤改善={mdd_improved}/{n}  累计收益改善={total_improved}/{n}")


if __name__ == "__main__":
    main()
