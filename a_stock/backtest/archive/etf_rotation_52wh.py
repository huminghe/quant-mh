"""
52周高点辅助信号验证（2026-07）

测试方向：
  1. 基线：风险调整动量（已知最优）
  2. 52W高点单独：score = price / rolling_max(252)
  3. 排名融合：(动量排名 + 52W排名) / 2 → 选Top3
  4. 乘积融合：动量得分 × 52W得分

对比基线（风险调整动量，夏普≈0.94），验证52W信号是否有增量价值。
注：不含拥挤度修正，目的是单独测试信号叠加效果，排除干扰。
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH       = 1_000_000
COMMISSION      = 0.0001
SLIPPAGE        = 0.0002
BENCHMARK       = "510300.SH"
START_DATE      = "2016-01-01"
IS_RATIO        = 0.8
MOMENTUM_WINDOW = 25
RISK_VOL_WINDOW = 21
W52_WINDOW      = 252   # 52周 ≈ 252交易日
TOP_N           = 3

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 信号计算 ───────────────────────────────────────────────

def momentum_score_single(prices: pd.Series) -> float:
    """OLS斜率×R²÷波动率（风险调整动量）"""
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_momentum_scores(close: pd.DataFrame) -> pd.DataFrame:
    """计算所有标的的风险调整动量得分"""
    scores = {}
    for code in close.columns:
        series = close[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score_single(series.iloc[i - MOMENTUM_WINDOW: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol  = rets.std() * np.sqrt(252)
                raw  = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close.index)


def calc_52wh_scores(close: pd.DataFrame) -> pd.DataFrame:
    """
    52周高点得分：price / rolling_max(252)
    值域 (0,1]，越接近1说明价格在历史高位，
    锚定效应产生持续动量（George & Hwang 2004）
    """
    scores = {}
    for code in close.columns:
        series = close[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(W52_WINDOW, len(series)):
            window_high = series.iloc[i - W52_WINDOW: i].max()
            ss.iloc[i] = series.iloc[i] / window_high if window_high > 0 else np.nan
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close.index)


def build_combined_rank(mom_scores: pd.DataFrame, w52_scores: pd.DataFrame) -> pd.DataFrame:
    """
    排名融合：两个信号各自排名后等权平均
    只对在两个信号都有有效值的标的做融合
    """
    combined = {}
    for date in mom_scores.index:
        m = mom_scores.loc[date].dropna()
        w = w52_scores.loc[date].dropna()
        common = m.index.intersection(w.index)
        if len(common) < 2:
            combined[date] = pd.Series(dtype=float)
            continue
        m_common = m[common]
        w_common = w[common]
        # 排名（升序=1，降序），归一化到[0,1]
        m_rank = m_common.rank(ascending=True) / len(common)
        w_rank = w_common.rank(ascending=True) / len(common)
        combined[date] = (m_rank + w_rank) / 2
    return pd.DataFrame(combined).T.reindex(mom_scores.index)


def build_product_score(mom_scores: pd.DataFrame, w52_scores: pd.DataFrame) -> pd.DataFrame:
    """
    乘积融合：动量得分 × 52W得分
    52W得分值域(0,1]，正动量×高52W = 既有趋势又在高位
    注：动量得分可以为负，负×正仍为负，不影响过滤逻辑
    """
    # 对齐两个矩阵
    common_idx  = mom_scores.index.intersection(w52_scores.index)
    common_cols = mom_scores.columns.intersection(w52_scores.columns)
    m = mom_scores.loc[common_idx, common_cols]
    w = w52_scores.loc[common_idx, common_cols]
    return m * w


# ── 回测引擎 ──────────────────────────────────────────────

def get_rebalance_dates(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_bt(close: pd.DataFrame, signal_scores: pd.DataFrame, rebal_dates: list) -> pd.Series:
    """通用回测引擎，根据传入的得分矩阵选Top3等权"""
    cash = INIT_CASH
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    for date in close.index:
        pv = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                pv += shares * close.loc[date, code]
        nav_series[date] = pv

        if date not in rebal_set:
            continue

        ds = signal_scores.loc[date].dropna() if date in signal_scores.index else pd.Series(dtype=float)
        # 只选正得分（空仓保护）
        ds_pos = ds[ds > 0]
        tc = list(ds_pos.nlargest(TOP_N).index)

        # 卖出不再持仓的标的
        for code in list(holdings.keys()):
            if code not in tc:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not tc:
            continue

        n = len(tc)
        weights = {c: 1.0 / n for c in tc}
        for code in tc:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            bp = price * (1 + SLIPPAGE / 2)
            tv = pv * weights[code]
            cs = holdings.get(code, 0)
            cv = cs * price
            diff = tv - cv
            if diff > bp * 100:
                bs = int(diff / bp / 100) * 100
                if bs > 0:
                    cost = bs * bp * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = cs + bs
            elif diff < -price * 100:
                ss = int(-diff / price / 100) * 100
                if ss > 0 and cs >= ss:
                    cash += ss * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = cs - ss

    return nav_series.dropna()


def calc_stats(nav: pd.Series, label: str = "") -> dict:
    rets  = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr  = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    wins   = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    losses = monthly[monthly < 0].abs().mean() if (monthly < 0).any() else 1
    return {
        "配置": label,
        "年化收益": f"{cagr*100:.1f}%",
        "夏普":     f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%",
        "Calmar":   f"{calmar:.2f}",
        "月胜率":   f"{win_rate:.1%}",
        "盈亏比":   f"{wins/losses:.2f}" if losses > 0 else "—",
        "_sharpe":  sharpe,
        "_maxdd":   max_dd,
        "_cagr":    cagr,
    }


# ── 主程序 ────────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= W52_WINDOW + MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
mom_scores = calc_momentum_scores(close)

print("计算52周高点得分...")
w52_scores = calc_52wh_scores(close)

print("计算融合信号...")
rank_combined = build_combined_rank(mom_scores, w52_scores)
prod_combined = build_product_score(mom_scores, w52_scores)

rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# IS/OOS 分割
n_days    = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
print(f"IS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")

# 各配置的信号
CONFIGS = [
    ("基线（风险调整动量）",   mom_scores),
    ("52W高点（单独）",        w52_scores),
    ("排名融合（等权）",       rank_combined),
    ("乘积融合",               prod_combined),
]

bench_nav = close[BENCHMARK].dropna()
bench_nav = bench_nav / bench_nav.iloc[0] * INIT_CASH

print("\n运行回测...")
navs_full = {}
navs_is   = {}
navs_oos  = {}

rebal_is  = [d for d in rebal_dates if d <  split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
close_is  = close[close.index <  split_date]
close_oos = close[close.index >= split_date]

for label, sig in CONFIGS:
    sig_is  = sig[sig.index <  split_date]  if sig is not None else None
    sig_oos = sig[sig.index >= split_date]  if sig is not None else None
    navs_full[label] = run_bt(close,     sig,     rebal_dates)
    navs_is[label]   = run_bt(close_is,  sig_is,  rebal_is)
    navs_oos[label]  = run_bt(close_oos, sig_oos, rebal_oos)
    s = calc_stats(navs_full[label])
    print(f"  {label:<22} 全样本夏普={s['_sharpe']:.3f}")

# ── 结果输出 ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("全样本全量指标（2016-2026）")
print("=" * 70)
rows = [calc_stats(navs_full[l], l) for l, _ in CONFIGS]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
df_out = df_out.set_index("配置")
print(df_out.to_string())

print("\n" + "=" * 70)
print("IS / OOS 验证")
print("=" * 70)
is_oos_rows = []
base_sharpe = calc_stats(navs_full["基线（风险调整动量）"])["_sharpe"]

for label, _ in CONFIGS:
    si = calc_stats(navs_is[label])
    so = calc_stats(navs_oos[label])
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    vs_base = calc_stats(navs_full[label])["_sharpe"] - base_sharpe
    status = "通过" if decay >= 0.5 else "警告:过拟合"
    print(f"\n{label}")
    print(f"  IS ：夏普={si['_sharpe']:.3f}  年化={si['年化收益']}  回撤={si['最大回撤']}")
    print(f"  OOS：夏普={so['_sharpe']:.3f}  年化={so['年化收益']}  回撤={so['最大回撤']}")
    print(f"  OOS/IS={decay:.2f}  [{status}]  全样本vs基线：{vs_base:+.3f}")
    is_oos_rows.append({"配置": label, "IS夏普": f"{si['_sharpe']:.3f}",
                        "OOS夏普": f"{so['_sharpe']:.3f}", "OOS/IS": f"{decay:.2f}",
                        "全样本vs基线": f"{vs_base:+.3f}", "状态": status})

print("\n汇总：")
print(pd.DataFrame(is_oos_rows).set_index("配置").to_string())

# ── 可视化 ────────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)

colors = {
    "基线（风险调整动量）": "#9E9E9E",
    "52W高点（单独）":      "#1565C0",
    "排名融合（等权）":     "#E53935",
    "乘积融合":             "#43A047",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 净值曲线
ax1 = axes[0]
for label, _ in CONFIGS:
    nav = navs_full[label]
    lw  = 2.0 if "基线" in label else 1.6
    ls  = "--" if "基线" in label else "-"
    ax1.plot(nav.index, nav / INIT_CASH, label=label,
             color=colors.get(label, "gray"), linewidth=lw, linestyle=ls)
ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800",
         linestyle=":", lw=1.0, alpha=0.6, label="沪深300")
ax1.axvline(split_date, color="red", linestyle="--", alpha=0.4, lw=1, label="IS/OOS分割")
ax1.set_title("52周高点辅助信号验证（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.3)

# 回撤对比
ax2 = axes[1]
for label, _ in CONFIGS:
    nav = navs_full[label]
    dd  = (nav - nav.cummax()) / nav.cummax() * 100
    s   = calc_stats(nav)
    lw  = 1.8 if "基线" in label else 1.4
    ax2.plot(dd.index, dd, label=f"{label}  MaxDD={dd.min():.1f}%",
             color=colors.get(label, "gray"), linewidth=lw)
ax2.set_ylabel("回撤(%)")
ax2.set_title("回撤对比")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "etf_rotation_52wh.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
