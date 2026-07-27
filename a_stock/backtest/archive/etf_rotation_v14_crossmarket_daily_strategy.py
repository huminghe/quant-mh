"""
第十一轮方向3深挖：跨市场溢出信号能否落地为可交易策略（含交易成本）

背景：v14_crossmarket_ic.py 已确认隔夜纳指收益 → 新能源车ETF(515330)次日收益
IC=+0.177，IS/OOS 均显著，是本轮所有候选里信号最强的。但这是日频信号，与当前
月度调仓框架（25日动量Top3）频率不匹配，直接套用无意义。这里单独测试：把它
做成一个独立的日频timing策略是否在扣除交易成本后仍然有效。

策略设计（最简单版本，避免过度设计）：
- 每个A股交易日T，用T-1日已知的隔夜纳指收益作为信号
- 隔夜收益 > 0：T日持有新能源车ETF（用开盘价买入，若已持有则不换手）
- 隔夜收益 <= 0：T日空仓（用开盘价卖出，若已空仓则不换手）
- 执行价用当日开盘价（隔夜信号在A股开盘前已知，完全可执行，不违反T+1——
  T+1限制的是"当日买入不可当日卖出"，这里是隔夜信号→次日开盘执行，不受影响）
- 交易成本按 trading-standards.md：单次完整回合(买+卖)约0.164%，这里按
  单边0.082%计（佣金万1+印花税千1×0.5仅ETF无印花税实际更低+滑点万2，
  ETF无印花税，单边成本≈佣金万1+滑点万2=万3=0.03%，保守用0.05%单边）

对比基准：新能源车ETF买入持有（同期）
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import init_pro

START_DATE = "2016-01-01"
END_DATE = "2026-07-10"
ETF_CODE = "515330.SH"
SINGLE_SIDE_COST = 0.0005  # 单边成本 0.05%（佣金万1+滑点万2，ETF无印花税，留余量）
RISK_FREE_ANNUAL = 0.02


def fetch_global_daily(pro, code: str) -> pd.Series:
    df = pro.index_global(ts_code=code, start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""))
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df["close"].pct_change().rename(code)


def load_ohlc(ts_code: str) -> pd.DataFrame:
    path = pathlib.Path(__file__).parent.parent.parent / "data" / "daily" / f"{ts_code}.parquet"
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")


print("拉取隔夜纳指收益 + 新能源车ETF开盘/收盘价...")
pro = init_pro()

ixic = fetch_global_daily(pro, "IXIC").dropna()
ohlc = load_ohlc(ETF_CODE)

# 信号：对每个A股交易日T，取最近一个 < T 的海外交易日的隔夜收益
signal = pd.Series(index=ohlc.index, dtype=float)
overnight_dates = ixic.index
for i, d in enumerate(ohlc.index):
    prior = overnight_dates[overnight_dates < d]
    if len(prior) == 0:
        continue
    signal.iloc[i] = ixic.loc[prior[-1]]

df = ohlc.copy()
df["signal"] = signal
df["position"] = (df["signal"] > 0).astype(float)  # 1=持有，0=空仓
df = df.dropna(subset=["signal"])

# ── 策略收益（用开盘价执行，当日开盘buy/sell，持有至次日开盘）─────────
# 当日收益 = position(T) × (close[T]/open[T] - 1)（T日开盘按信号建仓，持有到收盘）
df["day_ret"] = df["position"] * (df["close"] / df["open"] - 1)

# 换手：position 变化时产生一次买卖（开仓或平仓各算一次单边成本）
df["turnover"] = df["position"].diff().abs().fillna(df["position"].iloc[0])
df["cost"] = df["turnover"] * SINGLE_SIDE_COST
df["net_ret"] = df["day_ret"] - df["cost"]

# 基准：买入持有（open第一天到close最后一天，用close-to-close收益近似）
df["bh_ret"] = df["close"].pct_change().fillna(0)

trading_days = len(df)
n_years = trading_days / 252


def annualize(ret_series: pd.Series) -> dict:
    cum = (1 + ret_series).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_ANNUAL) / ann_vol if ann_vol > 1e-9 else np.nan
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    win_rate = (ret_series > 0).mean()
    return {"年化收益": ann_ret, "年化波动": ann_vol, "夏普": sharpe, "最大回撤": mdd, "日胜率": win_rate}


print("\n" + "=" * 80)
print(f"全样本（{df.index[0].date()} ~ {df.index[-1].date()}，{trading_days}个交易日）")
print("=" * 80)

strat_stats = annualize(df["net_ret"])
bh_stats = annualize(df["bh_ret"])
avg_turnover_per_year = df["turnover"].sum() / n_years

print(f"策略（含成本）：{strat_stats}")
print(f"年均换手次数（单边）：{avg_turnover_per_year:.0f} 次/年")
print(f"年化成本拖累：{(df['cost'].sum() / n_years):.2%}")
print(f"基准买入持有：{bh_stats}")

# ── IS/OOS 分段（与项目惯例一致）─────────────────────────
print("\n" + "=" * 80)
print("IS（2016-2024.01）/ OOS（2024.02-2026.07）")
print("=" * 80)
is_df = df[df.index < "2024-02-01"]
oos_df = df[df.index >= "2024-02-01"]
is_n_years = len(is_df) / 252
oos_n_years = len(oos_df) / 252


def annualize_n(ret_series, ny):
    cum = (1 + ret_series).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / ny) - 1
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_ANNUAL) / ann_vol if ann_vol > 1e-9 else np.nan
    return ann_ret, sharpe


is_ret, is_sharpe = annualize_n(is_df["net_ret"], is_n_years)
oos_ret, oos_sharpe = annualize_n(oos_df["net_ret"], oos_n_years)
print(f"IS：年化收益={is_ret:+.2%}，夏普={is_sharpe:.3f}")
print(f"OOS：年化收益={oos_ret:+.2%}，夏普={oos_sharpe:.3f}")

# ── 结论 ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"策略夏普={strat_stats['夏普']:.3f} vs 基准买入持有夏普={bh_stats['夏普']:.3f}")
print(f"年均单边换手{avg_turnover_per_year:.0f}次，年化成本拖累{(df['cost'].sum() / n_years):.2%}")
if strat_stats["夏普"] > bh_stats["夏普"] and oos_sharpe > is_sharpe * 0.5:
    print("判定：扣除成本后仍优于买入持有，且OOS未大幅衰减，值得进一步验证与其他标的的组合效果。")
else:
    print("判定：扣除日频换手成本后信号价值被侵蚀，或OOS衰减严重，不建议作为独立策略上线。")

print("\n完成。")
