"""
第十一轮方向4：成长/价值风格轮动可预测性 IC 检验

背景：已测方向全部是"行业间"轮动（半导体vs银行vs消费），从未测过"风格间"
轮动（成长vs价值）。国证成长100(399372)/国证价值100(399373)两个风格指数
2015年即有数据，覆盖完整回测窗口。若风格轮动本身可预测，可以作为行业轮动
的补充维度（例如同一行业下选成长/价值特征更匹配当前风格周期的ETF）。

方法（先IC检验排除法）：
1. 计算成长-价值相对强弱（成长/价值净值比值）的动量，检验其能否预测未来
   1个月的成长-价值相对收益（即风格切换本身是否有惯性/可预测性）
2. 检验该信号与现有风险调整动量评分的相关性（若两者高度相关，说明是
   同一信息的重复表达）
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import init_pro

START_DATE = "2015-01-01"
END_DATE = "2026-07-10"
MOMENTUM_WINDOW = 25


def fetch_index_daily(pro, ts_code: str) -> pd.Series:
    df = pro.index_daily(ts_code=ts_code, start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""))
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").set_index("trade_date")["close"]


print("从 tushare 拉取国证成长100/价值100指数数据...")
pro = init_pro()

growth = fetch_index_daily(pro, "399372.SZ")
value = fetch_index_daily(pro, "399373.SZ")
print(f"成长100：{len(growth)} 个交易日，{growth.index[0].date()} ~ {growth.index[-1].date()}")
print(f"价值100：{len(value)} 个交易日，{value.index[0].date()} ~ {value.index[-1].date()}")

df = pd.DataFrame({"growth": growth, "value": value}).dropna()
df["gv_ratio"] = df["growth"] / df["value"]

# ── 诊断1：风格相对强弱的动量能否预测未来1月的风格相对收益 ─────────
print("\n" + "=" * 80)
print("诊断1：成长/价值相对强弱动量 vs 未来1月相对收益 IC 检验")
print("=" * 80)

df["gv_mom"] = df["gv_ratio"].pct_change(MOMENTUM_WINDOW)
df["gv_fwd_1m"] = df["gv_ratio"].pct_change(21).shift(-21)

test_df = df[["gv_mom", "gv_fwd_1m"]].dropna()
ic = test_df["gv_mom"].corr(test_df["gv_fwd_1m"])
hit = (np.sign(test_df["gv_mom"]) == np.sign(test_df["gv_fwd_1m"])).mean()

yearly_ics = []
for yr in sorted(set(test_df.index.year)):
    seg = test_df[test_df.index.year == yr]
    if len(seg) < 20:
        continue
    yearly_ics.append((yr, seg["gv_mom"].corr(seg["gv_fwd_1m"])))
pos_ratio = np.mean([1 if v > 0 else 0 for _, v in yearly_ics]) if yearly_ics else np.nan

print(f"25日风格动量 IC={ic:+.4f}，方向命中率={hit:.1%}，样本数={len(test_df)}")
print(f"年度IC：{[(y, round(v, 3)) for y, v in yearly_ics]}")
print(f"年度同向占比：{pos_ratio:.1%}")

# ── 诊断2：不同窗口扫描（避免25日窗口是唯一有效点的偶然性）─────────
print("\n" + "=" * 80)
print("诊断2：多窗口扫描（10/25/42/63/126日）")
print("=" * 80)
for w in [10, 25, 42, 63, 126]:
    mom = df["gv_ratio"].pct_change(w)
    test_w = pd.DataFrame({"mom": mom, "fwd": df["gv_fwd_1m"]}).dropna()
    ic_w = test_w["mom"].corr(test_w["fwd"])
    print(f"  窗口={w}日：IC={ic_w:+.4f}，样本数={len(test_w)}")

# ── 诊断3：风格切换是否与市场整体波动率相关（同源检验，类比"高波动期趋势最强"）
print("\n" + "=" * 80)
print("诊断3：风格反转 vs 均值回归特征 —— 用自相关判断惯性还是反转")
print("=" * 80)
autocorr_1m = df["gv_fwd_1m"].autocorr(lag=21)
print(f"成长-价值相对收益 21日自相关（滞后1个月）={autocorr_1m:+.4f}")
print("（正值支持惯性/趋势延续，负值支持均值回归/风格快速切换）")

# ── 结论 ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"25日窗口IC={ic:+.4f}（阈值0.03）")
if abs(ic) < 0.03:
    print("判定：IC绝对值 < 0.03，信号强度不足，不建议进入组合层面回测。")
elif pos_ratio < 0.6:
    print(f"判定：年度同向占比{pos_ratio:.1%} < 60%，方向不稳定，不建议进入组合层面回测。")
else:
    print("判定：IC强度、方向一致性通过初筛，值得进入组合层面回测验证。")

print("\n完成。")
