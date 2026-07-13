"""
第十二轮方向1：行业收益离散度(Dispersion) regime 信号 —— 诊断性分析

背景：网络调研发现的新候选（美股证据为主，SSGA/Morningstar/S&P Dow Jones
2025年专题），核心思想是横截面收益离散度可以作为"该集中持仓还是该分散
持仓"的元信号：离散度高时，赢家跑赢输家的幅度大，集中（Top3）更有优势；
离散度低时，标的表现趋同，集中持仓承担了额外的选股风险却拿不到额外收益，
分散（等权全池）更稳。国内暂无实证，纯自有价格数据可算，先做诊断，不
直接设计参数网格（按项目惯例避免过拟合）。

方法：
1. 在每个调仓日，用trailing MOMENTUM_WINDOW日收益计算全池横截面离散度
   （标准差），按月度分布分三档（低/中/高）。
2. 对每个调仓日，比较"当月Top3组合前瞻收益" vs "当月等权全池前瞻收益"，
   得到超额（Top3 - 等权）。
3. 检验：离散度分档 与 Top3超额 是否存在正相关（IC检验），以及三档
   分组下Top3超额的均值差异。若无显著关系，直接排除，不进一步做
   "根据离散度切换集中/分散"的独立策略回测。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix
from etf_universe import ETF_UNIVERSE

START_DATE = "2016-01-01"
MOMENTUM_WINDOW = 25
RISK_VOL_WINDOW = 21
TOP_N = 3


def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_risk_adj_momentum(close_matrix: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score(series.iloc[i - MOMENTUM_WINDOW: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close_matrix.index)


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = MOMENTUM_WINDOW + 20
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

daily_rets = close.pct_change()
fwd_1m = close.pct_change().rolling(21).sum().shift(-21)

print("计算风险调整动量得分...")
scores = calc_risk_adj_momentum(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# ── 1. 计算每个调仓日的横截面离散度（trailing窗口收益的标准差）────

print("\n" + "=" * 80)
print("诊断1：横截面离散度计算（trailing 25日收益标准差）")
print("=" * 80)

trailing_ret = close.pct_change(MOMENTUM_WINDOW)
dispersion = trailing_ret.std(axis=1)

rows = []
for d in rebal_dates:
    if d not in dispersion.index:
        continue
    disp = dispersion.loc[d]
    if pd.isna(disp):
        continue

    day_scores = scores.loc[d].dropna()
    pos_scores = day_scores[day_scores > 0].nlargest(TOP_N * 3)
    top_codes = list(pos_scores.index)[:TOP_N]

    fwd = fwd_1m.loc[d]
    if not top_codes or fwd[top_codes].isna().all():
        continue
    top3_fwd = fwd[top_codes].mean()

    all_valid = fwd.dropna()
    if len(all_valid) < 10:
        continue
    equal_fwd = all_valid.mean()

    rows.append({"date": d, "dispersion": disp, "top3_fwd": top3_fwd,
                 "equal_fwd": equal_fwd, "excess": top3_fwd - equal_fwd})

df = pd.DataFrame(rows).dropna()
print(f"有效调仓月份：{len(df)} 个")

# ── 2. IC检验：离散度 vs Top3超额收益 ─────────────────────────

print("\n" + "=" * 80)
print("诊断2：离散度 vs (Top3 - 等权全池) 前瞻收益超额 —— IC检验")
print("=" * 80)

ic = df["dispersion"].corr(df["excess"])
hit_rate = (np.sign(df["dispersion"] - df["dispersion"].median()) ==
            np.sign(df["excess"] - df["excess"].median())).mean()
print(f"IC（离散度 vs Top3超额）= {ic:+.3f}")
print(f"高低离散度分组与超额高低分组一致率 = {hit_rate:.1%}")

# 年度拆分，检查方向稳定性
df["year"] = df["date"].dt.year
yearly_ic = df.groupby("year").apply(
    lambda g: g["dispersion"].corr(g["excess"]) if len(g) >= 6 else np.nan
)
print("\n年度IC（样本<6个月的年份跳过）：")
print(yearly_ic.dropna().to_string())
same_sign_ratio = (np.sign(yearly_ic.dropna()) == np.sign(ic)).mean() if ic != 0 else 0
print(f"年度同向占比：{same_sign_ratio:.1%}")

# ── 3. 三档分组对比 ──────────────────────────────────────────

print("\n" + "=" * 80)
print("诊断3：离散度三档分组 —— Top3超额收益对比")
print("=" * 80)

df["tercile"] = pd.qcut(df["dispersion"], 3, labels=["低离散度", "中离散度", "高离散度"])
grp = df.groupby("tercile")["excess"].agg(["mean", "median", "std", "count"])
grp.columns = ["超额均值", "超额中位数", "超额std", "样本数"]
print(grp.to_string())

low_excess = df[df["tercile"] == "低离散度"]["excess"].mean()
high_excess = df[df["tercile"] == "高离散度"]["excess"].mean()
print(f"\n高离散度档 - 低离散度档 超额均值差 = {high_excess - low_excess:+.4f}")

# ── 结论 ─────────────────────────────────────────────────

print("\n" + "=" * 80)
if abs(ic) < 0.03 or same_sign_ratio < 0.6:
    print(f"结论：IC={ic:+.3f}，年度同向占比={same_sign_ratio:.1%}，"
          f"未达到项目排除阈值（|IC|>=0.03 且年度同向占比>=60%），判定为噪音，排除。")
else:
    print(f"结论：IC={ic:+.3f}，年度同向占比={same_sign_ratio:.1%}，达到信号质量阈值，"
          f"可考虑进入组合层面回测（用离散度分档切换Top3/等权全池）。")
print("=" * 80)
