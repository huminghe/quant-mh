"""
第十一轮方向1：滚动收益偏度因子 IC 检验

背景：已测的仓位管理方向全部停留在二阶矩（方差/协方差，如Ledoit-Wolf/组合波动率
目标/拥挤度衰减加权），从未探索三阶矩（偏度）。国内实证（中信一级行业指数
2006-2019）显示行业收益偏度分层有效，与波动率因子存在互补而非完全重叠的信息。

方法（按项目惯例，先IC检验排除法，不直接进组合回测就测出负效果）：
1. 计算滚动N日收益偏度（N=21/42/63三档），与未来1月收益做IC检验
2. 检验偏度因子是否与当前风险调整动量评分高度相关（若相关性极高，说明是
   同一信息的重复表达，不值得单独引入）
3. IC若显著（|IC|>0.03 且方向稳定），再决定是否进入组合层面回测；
   若不显著，直接排除，不浪费时间写组合回测代码
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix
from etf_universe import ETF_UNIVERSE

START_DATE = "2016-01-01"
MOMENTUM_WINDOW = 25
RISK_VOL_WINDOW = 21
SKEW_WINDOWS = [21, 42, 63]


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


print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = max(SKEW_WINDOWS) + MOMENTUM_WINDOW + 20
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

daily_rets = close.pct_change()
fwd_1m = close.pct_change().rolling(21).sum().shift(-21)

print("\n计算风险调整动量得分（作为对照，检查与偏度因子的相关性）...")
mom_scores = calc_risk_adj_momentum(close)

# ── IC 检验：滚动偏度 vs 未来1月收益 ──────────────────────────

print("\n" + "=" * 80)
print("诊断1：滚动收益偏度因子 IC 检验（信号 vs 未来1月收益）")
print("=" * 80)

ic_summary = []
skew_scores_by_window = {}
for w in SKEW_WINDOWS:
    skew = daily_rets.rolling(w).skew()
    skew_scores_by_window[w] = skew

    ics, hits = [], []
    for code in valid_codes:
        sig = skew[code].dropna()
        fwd = fwd_1m[code].dropna()
        df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
        if len(df) < 100:
            continue
        ic = df["sig"].corr(df["fwd"])
        hit = (np.sign(df["sig"]) == np.sign(df["fwd"])).mean()
        ics.append(ic)
        hits.append(hit)

    # 分年度IC，检查方向稳定性
    yearly_ics = []
    for yr in sorted(set(close.index.year)):
        yr_ic_list = []
        for code in valid_codes:
            sig = skew[code].dropna()
            fwd = fwd_1m[code].dropna()
            df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
            df = df[(df.index.year == yr)]
            if len(df) < 20:
                continue
            yr_ic_list.append(df["sig"].corr(df["fwd"]))
        if yr_ic_list:
            yearly_ics.append((yr, np.nanmean(yr_ic_list)))

    pos_year_ratio = np.mean([1 if v > 0 else 0 for _, v in yearly_ics]) if yearly_ics else np.nan

    ic_summary.append({
        "偏度窗口": f"{w}日",
        "标的数": len(ics),
        "平均IC": np.mean(ics),
        "IC中位数": np.median(ics),
        "平均方向命中率": np.mean(hits),
        "年度IC同向占比": pos_year_ratio,
    })

ic_df = pd.DataFrame(ic_summary).set_index("偏度窗口")
print(ic_df.to_string())

print("\n各标的单独IC（63日偏度窗口，供抽查）：")
skew_63 = skew_scores_by_window[63]
detail_rows = []
for code in valid_codes:
    sig = skew_63[code].dropna()
    fwd = fwd_1m[code].dropna()
    df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
    if len(df) < 100:
        continue
    ic = df["sig"].corr(df["fwd"])
    detail_rows.append({"代码": code, "名称": ETF_UNIVERSE.get(code, code), "IC": ic, "样本数": len(df)})
detail_df = pd.DataFrame(detail_rows).sort_values("IC")
print(detail_df.to_string(index=False))

# ── 诊断2：偏度因子与风险调整动量的相关性（是否是同一信息的重复表达）──

print("\n" + "=" * 80)
print("诊断2：63日偏度因子 与 风险调整动量评分 的截面相关性")
print("=" * 80)

corrs = []
common_dates = mom_scores.index.intersection(skew_63.index)
for date in common_dates[::21]:  # 每21个交易日采样一次，减少计算量且近似月度频率
    mom_row = mom_scores.loc[date].dropna()
    skew_row = skew_63.loc[date].dropna()
    common_codes = mom_row.index.intersection(skew_row.index)
    if len(common_codes) < 10:
        continue
    corrs.append(mom_row[common_codes].corr(skew_row[common_codes]))

corrs = pd.Series(corrs).dropna()
print(f"截面相关系数（动量评分 vs 63日偏度）：均值={corrs.mean():.3f}，"
      f"中位数={corrs.median():.3f}，std={corrs.std():.3f}，样本数={len(corrs)}")

# ── 结论 ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
best_w = ic_df["平均IC"].abs().idxmax()
best_ic = ic_df.loc[best_w, "平均IC"]
best_year_ratio = ic_df.loc[best_w, "年度IC同向占比"]
print(f"最强IC窗口：{best_w}，平均IC={best_ic:+.4f}，年度同向占比={best_year_ratio:.1%}")
print(f"与风险调整动量截面相关性：{corrs.mean():+.3f}")

if abs(best_ic) < 0.03:
    print("判定：IC绝对值 < 0.03，信号强度不足，不建议进入组合层面回测。")
elif best_year_ratio < 0.6:
    print("判定：IC方向年度不稳定（同向占比<60%），信号不可靠，不建议进入组合层面回测。")
elif abs(corrs.mean()) > 0.5:
    print("判定：与已有动量评分高度相关（|corr|>0.5），大概率是同一信息的重复表达，价值有限。")
else:
    print("判定：IC强度、方向稳定性、与现有信号的独立性均通过初筛，值得进入组合层面回测验证。")

print("\n完成。")
