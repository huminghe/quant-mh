"""
资金费率异常监控：标的市场结构突变信号（不是趋势预测）

背景：2026-10-21/23调研"标的本身结构性变化"监控方向。资金费率反映永续合约
多空力量失衡程度，极端持续偏离通常对应挤仓/异常投机情绪，是市场结构层面的
信号，不是价格趋势信号——这与已上线的 FR_MA7>0 空头过滤（判断方向）是不同
用途：本脚本只做"异常检测"，不做方向判断。

覆盖标的：BTC/ETH/SOL/DOGE（当前8个实盘策略的4个底层标的）。预期结论（调研
已提前判断）：对BTC意义有限（流动性最深，费率结构最稳定），对DOGE这类山寨
币更有参考价值（费率波动幅度更大，极端值更容易对应真实的结构性事件，比如
散户FOMO挤仓）。本脚本用全部4个标的的真实数据验证这个预期是否成立，不是
预先假设BTC可以跳过。

方法：
  - 用 ccxt.binance fetchFundingRateHistory 拉历史资金费率（8小时结算一次）
  - 滚动窗口（默认90天=270期）算每个标的费率的均值/标准差，算当前值的z-score
  - 异常判定用两级：
    1. 单期极值：|z-score| > 阈值（默认3）
    2. 持续偏离：连续N期（默认3期=1天）同向z-score > 阈值的一半，比单次尖刺
       更有意义（单次尖刺可能是瞬时清算，连续偏离更接近结构性失衡）
  - 不与策略PnL关联，这是纯市场结构层监控，不是仓位决策输入

用法：
  python funding_rate_monitor.py                  # 默认跑全部4个标的，近90天
  python funding_rate_monitor.py --days 180        # 拉更长历史看基线是否稳定
  python funding_rate_monitor.py --asset DOGE      # 只看单个标的明细
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import ccxt

ASSETS = ["BTC", "ETH", "SOL", "DOGE"]
SYMBOL_MAP = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "DOGE": "DOGE/USDT"}

Z_THRESHOLD = 3.0          # 单期极值 z-score 阈值
PERSIST_PERIODS = 3        # 连续偏离期数（3期=1天，8小时结算一次）
PERSIST_Z_THRESHOLD = 1.5  # 持续偏离的 z-score 阈值（比单期阈值宽松，但要求连续）


def fetch_funding_history(symbol: str, days: int) -> list[dict]:
    ex = ccxt.binance({"options": {"defaultType": "future"}, "timeout": 30000})
    since = ex.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rates = []
    while True:
        for attempt in range(3):
            try:
                rates = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        if not rates:
            break
        all_rates.extend(rates)
        if len(rates) < 1000:
            break
        since = rates[-1]["timestamp"] + 1
    return all_rates


def analyze_asset(asset: str, days: int) -> dict:
    symbol = SYMBOL_MAP[asset]
    print(f"  拉取 {symbol} 资金费率历史（近{days}天）...", end=" ", flush=True)
    rates = fetch_funding_history(symbol, days)
    print(f"done ({len(rates)}期)")

    values = np.array([r["fundingRate"] for r in rates])
    timestamps = [r["datetime"] for r in rates]

    mean, std = values.mean(), values.std(ddof=1)
    z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)

    single_spikes = np.where(np.abs(z_scores) > Z_THRESHOLD)[0]

    persist_events = []
    i = 0
    while i < len(z_scores) - PERSIST_PERIODS + 1:
        window = z_scores[i:i + PERSIST_PERIODS]
        if np.all(window > PERSIST_Z_THRESHOLD) or np.all(window < -PERSIST_Z_THRESHOLD):
            persist_events.append(i)
            i += PERSIST_PERIODS  # 跳过已计入的窗口，避免重叠计数同一段
        else:
            i += 1

    return {
        "asset": asset,
        "n": len(values),
        "mean": mean,
        "std": std,
        "min": values.min(),
        "max": values.max(),
        "single_spikes": [(timestamps[idx], values[idx], z_scores[idx]) for idx in single_spikes],
        "persist_events": [(timestamps[idx], z_scores[idx:idx + PERSIST_PERIODS].mean()) for idx in persist_events],
        "latest_z": z_scores[-1] if len(z_scores) else None,
        "latest_rate": values[-1] if len(values) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--asset", choices=ASSETS, default=None)
    args = parser.parse_args()

    targets = [args.asset] if args.asset else ASSETS

    print(f"===== 资金费率异常监控（近{args.days}天，单期z阈值={Z_THRESHOLD}，"
          f"持续{PERSIST_PERIODS}期z阈值={PERSIST_Z_THRESHOLD}） =====\n")

    results = []
    for asset in targets:
        r = analyze_asset(asset, args.days)
        results.append(r)

        print(f"--- {r['asset']} ---")
        print(f"    样本数={r['n']}  均值={r['mean']*100:.4f}%  标准差={r['std']*100:.4f}%  "
              f"区间=[{r['min']*100:.4f}%, {r['max']*100:.4f}%]")
        print(f"    最新一期：rate={r['latest_rate']*100:.4f}%  z-score={r['latest_z']:+.2f}")
        print(f"    单期极值事件：{len(r['single_spikes'])}次")
        for ts, rate, z in r["single_spikes"][-5:]:
            print(f"      {ts}  rate={rate*100:+.4f}%  z={z:+.2f}")
        print(f"    持续偏离事件（连续{PERSIST_PERIODS}期）：{len(r['persist_events'])}次")
        for ts, avg_z in r["persist_events"][-5:]:
            print(f"      {ts}起  平均z={avg_z:+.2f}")
        print()

    print("===== 跨标的对比（验证'对BTC意义有限，对山寨币更有价值'的预期） =====")
    print(f"{'标的':<8}{'费率标准差%':<14}{'单期极值次数':<14}{'持续偏离次数'}")
    for r in results:
        print(f"{r['asset']:<8}{r['std']*100:<14.4f}{len(r['single_spikes']):<14}{len(r['persist_events'])}")


if __name__ == "__main__":
    main()
