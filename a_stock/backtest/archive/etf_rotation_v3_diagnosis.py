"""
v3 回测结果诊断脚本
1. 方向C 区制序列分析：按年统计牛/熊/震荡分布，输出具体熊市判断时段
2. 方向A+C IS期崩溃原因：逐年分析 IS 期（2016-2024）净值曲线，定位问题年份
3. 方向A+D 回撤压缩来源：分析禁买期的时间分布，确认效果来源
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
from fetch_data import load_close_matrix, init_pro

# ── 参数（与 v3 保持一致）─────────────────────────────────
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
BENCHMARK        = "510300.SH"
BENCHMARK_INDEX  = "000300.SH"
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21
PE_WINDOW        = 252
VOL_WINDOW_C     = 20
NORTHBOUND_MA    = 20

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 数据加载（复用 v3 逻辑）──────────────────────────────

def load_pe_data() -> pd.Series:
    pro = init_pro()
    df = pro.index_dailybasic(
        ts_code=BENCHMARK_INDEX, start_date="20150101", end_date="20261231",
        fields="trade_date,pe_ttm",
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").set_index("trade_date")["pe_ttm"]


def load_northbound_data() -> pd.Series:
    import akshare as ak
    df = ak.stock_hsgt_hist_em(symbol="沪股通")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").set_index("日期")
    return df["当日成交净买额"].astype(float)


def build_regime(close, pe_series, pe_window=PE_WINDOW, vol_window=VOL_WINDOW_C):
    pe = pe_series.reindex(close.index).ffill()
    if BENCHMARK in close.columns:
        bench_ret = close[BENCHMARK].pct_change()
    else:
        bench_ret = pd.Series(dtype=float, index=close.index)
    vol_20 = bench_ret.rolling(vol_window).std() * np.sqrt(252)
    regime = pd.Series(3, index=close.index, dtype=int)
    pe_pct_series  = pd.Series(np.nan, index=close.index)
    vol_pct_series = pd.Series(np.nan, index=close.index)
    for i in range(pe_window, len(close.index)):
        date = close.index[i]
        curr_pe  = pe.iloc[i]
        curr_vol = vol_20.iloc[i]
        if pd.isna(curr_pe) or pd.isna(curr_vol):
            continue
        hist_pe  = pe.iloc[i - pe_window: i].dropna()
        pe_pct   = (hist_pe  < curr_pe).mean()  if len(hist_pe)  >= 50 else 0.5
        hist_vol = vol_20.iloc[i - pe_window: i].dropna()
        vol_pct  = (hist_vol < curr_vol).mean() if len(hist_vol) >= 50 else 0.5
        pe_pct_series[date]  = pe_pct
        vol_pct_series[date] = vol_pct
        is_bear = pe_pct > 0.80 or vol_pct > 0.80
        is_bull = pe_pct < 0.60 and vol_pct < 0.60
        if is_bear:
            regime[date] = 1
        elif is_bull:
            regime[date] = 3
        else:
            regime[date] = 2
    return regime, pe_pct_series, vol_pct_series


def build_northbound_filter(close, nb_series, ma_window=NORTHBOUND_MA):
    nb_ma = nb_series.rolling(ma_window).mean()
    nb_ma_aligned = nb_ma.reindex(close.index).ffill()
    allow_buy = nb_ma_aligned >= 0
    return allow_buy.fillna(True)


def momentum_score(prices):
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix, window=MOMENTUM_WINDOW, risk_adj=True, risk_vol_window=RISK_VOL_WINDOW):
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            pw  = series.iloc[i - window: i]
            raw = momentum_score(pw)
            if risk_adj and i >= risk_vol_window:
                rets = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol  = rets.std() * np.sqrt(252)
                raw  = raw / vol if vol > 1e-6 else raw
            score_series.iloc[i] = raw
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


def get_rebalance_dates(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_backtest(close, scores, rebal_dates, top_n=TOP_N,
                 init_cash=INIT_CASH, use_ledoit_wolf=False,
                 top_n_regime=None, allow_buy_filter=None):
    if use_ledoit_wolf:
        from sklearn.covariance import LedoitWolf
    cash = init_cash
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)
    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value
        if date not in rebal_set:
            continue
        day_scores = scores.loc[date].dropna()
        effective_n = int(top_n_regime[date]) if (top_n_regime is not None and date in top_n_regime.index) else top_n
        pos_scores  = day_scores[day_scores > 0].nlargest(effective_n * 3)
        candidates  = list(pos_scores.index)
        target_codes = candidates[:effective_n]
        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE/2) * (1 - COMMISSION)
                del holdings[code]
        if not target_codes:
            continue
        nb_allow = bool(allow_buy_filter.get(date, True)) if allow_buy_filter is not None else True
        n = len(target_codes)
        if use_ledoit_wolf and n >= 2 and nb_allow:
            date_loc  = close.index.get_loc(date)
            hist_start = max(0, date_loc - 252)
            ret_hist  = close[target_codes].iloc[hist_start:date_loc].pct_change().dropna()
            if len(ret_hist) >= 60:
                try:
                    lw    = LedoitWolf().fit(ret_hist.values)
                    cov   = lw.covariance_
                    ones  = np.ones(n)
                    inv_cov = np.linalg.pinv(cov)
                    raw_w = inv_cov @ ones
                    raw_w = np.clip(raw_w, 0.05, None)
                    raw_w = np.clip(raw_w, None, 0.70 * raw_w.sum())
                    w_arr = raw_w / raw_w.sum()
                    weights = {code: float(w_arr[i]) for i, code in enumerate(target_codes)}
                except Exception:
                    weights = {c: 1.0/n for c in target_codes}
            else:
                weights = {c: 1.0/n for c in target_codes}
        else:
            weights = {c: 1.0/n for c in target_codes}
        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            if not nb_allow and code not in holdings:
                continue
            buy_price = price * (1 + SLIPPAGE/2)
            target_value   = port_value * weights[code]
            current_shares = holdings.get(code, 0)
            current_value  = current_shares * price
            diff = target_value - current_value
            if diff > buy_price * 100:
                buy_shares = int(diff / buy_price / 100) * 100
                if buy_shares > 0:
                    cost = buy_shares * buy_price * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = current_shares + buy_shares
            elif diff < -price * 100:
                sell_shares = int(-diff / price / 100) * 100
                if sell_shares > 0 and current_shares >= sell_shares:
                    cash += sell_shares * price * (1 - SLIPPAGE/2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares
    return nav_series.dropna()


def calc_stats(nav):
    rets  = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr  = (nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd}


# ═══════════════════════════════════════════════════════
print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]

pe_series = load_pe_data()
nb_series = load_northbound_data()

print("预计算信号...")
scores   = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

regime, pe_pct_s, vol_pct_s = build_regime(close, pe_series)
nb_filter = build_northbound_filter(close, nb_series)

n_days     = len(close)
split_idx  = int(n_days * IS_RATIO)
split_date = close.index[split_idx]

out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════
# 诊断 1：区制序列逐年分布
# ══════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("诊断1：方向C 区制序列逐年分布")
print("=" * 65)

regime_df = pd.DataFrame({
    "regime": regime,
    "pe_pct": pe_pct_s,
    "vol_pct": vol_pct_s,
})
regime_df["year"] = regime_df.index.year

label_map = {1: "熊市(Top1)", 2: "震荡(Top2)", 3: "牛市(Top3)"}
print(f"\n{'年份':<6}  {'牛市Top3':>10}  {'震荡Top2':>10}  {'熊市Top1':>10}")
for yr, grp in regime_df.groupby("year"):
    n_total = len(grp)
    bull  = (grp["regime"] == 3).sum() / n_total
    range_ = (grp["regime"] == 2).sum() / n_total
    bear  = (grp["regime"] == 1).sum() / n_total
    print(f"{yr:<6}  {bull:>9.1%}  {range_:>10.1%}  {bear:>10.1%}")

# 哪些月份是"纯熊市"（当月全部判为熊市）
print("\n\n完全被判为熊市的月份（调仓日区制=1）：")
rebal_regime = pd.Series({d: regime.get(d, 3) for d in rebal_dates})
bear_rebal_dates = [d for d in rebal_dates if regime.get(d, 3) == 1]
print(f"调仓日中熊市占比：{len(bear_rebal_dates)}/{len(rebal_dates)} = {len(bear_rebal_dates)/len(rebal_dates):.1%}")
print("熊市调仓日：", [str(d.date()) for d in bear_rebal_dates])

# ══════════════════════════════════════════════════════
# 诊断 2：A+C IS 期崩溃——逐年净值
# ══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("诊断2：方向A+C（LW+区制）IS 期逐年净值")
print("=" * 65)

close_is = close[close.index < split_date]
sc_is    = scores[scores.index < split_date]
reg_is   = regime[regime.index < split_date]
rebal_is = [d for d in rebal_dates if d < split_date]

nav_ac_is   = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=True,  top_n_regime=reg_is)
nav_base_is = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=False, top_n_regime=None)

print(f"\n{'年份':<6}  {'基线NAV末':>10}  {'A+C NAV末':>10}  {'基线年化':>10}  {'A+C年化':>10}")
for yr in range(2016, split_date.year + 1):
    yr_end  = pd.Timestamp(f"{yr}-12-31")
    yr_start = pd.Timestamp(f"{yr}-01-01")
    b_yr = nav_base_is[(nav_base_is.index >= yr_start) & (nav_base_is.index <= yr_end)]
    a_yr = nav_ac_is[(nav_ac_is.index  >= yr_start) & (nav_ac_is.index  <= yr_end)]
    if b_yr.empty or a_yr.empty:
        continue
    b_ret = b_yr.iloc[-1] / b_yr.iloc[0] - 1
    a_ret = a_yr.iloc[-1] / a_yr.iloc[0] - 1
    print(f"{yr:<6}  {b_yr.iloc[-1]:>10.0f}  {a_yr.iloc[-1]:>10.0f}  {b_ret:>9.1%}  {a_ret:>9.1%}")

# ══════════════════════════════════════════════════════
# 诊断 3：方向A+D 回撤压缩来源——北向禁买时段分析
# ══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("诊断3：方向A+D（LW+北向） 北向禁买时段分析")
print("=" * 65)

nb_ma = nb_series.rolling(NORTHBOUND_MA).mean().reindex(close.index).ffill()
forbid_dates = close.index[nb_ma < 0]
allow_dates  = close.index[nb_ma >= 0]
print(f"\n禁买日共 {len(forbid_dates)} 天（占 {len(forbid_dates)/len(close):.1%}），允许日 {len(allow_dates)} 天")

# 连续禁买段
forbid_mask = nb_ma < 0
segments = []
in_seg, seg_start = False, None
for date in close.index:
    if forbid_mask.get(date, False):
        if not in_seg:
            in_seg, seg_start = True, date
    else:
        if in_seg:
            segments.append((seg_start, date))
            in_seg = False
if in_seg:
    segments.append((seg_start, close.index[-1]))

print(f"\n连续禁买段（共 {len(segments)} 段），列出最长的10段：")
seg_df = pd.DataFrame(segments, columns=["start", "end"])
seg_df["days"] = (seg_df["end"] - seg_df["start"]).dt.days
seg_df = seg_df.sort_values("days", ascending=False).reset_index(drop=True)
print(seg_df.head(10).to_string(index=False))

# 禁买期内基线策略的表现（用来判断"禁买"是否真的避开了下跌）
nav_base = run_backtest(close, scores, rebal_dates)
nav_ad   = run_backtest(close, scores, rebal_dates, use_ledoit_wolf=True, allow_buy_filter=nb_filter)

print("\n\n禁买期内基线与A+D的每日净值对比（最长5段内的表现）：")
print(f"{'段':>3}  {'时段':>24}  {'基线收益':>10}  {'A+D收益':>10}  {'基线最大回撤':>12}")
for i, row in seg_df.head(5).iterrows():
    s, e = row["start"], row["end"]
    b_seg = nav_base[(nav_base.index >= s) & (nav_base.index <= e)]
    a_seg = nav_ad[(nav_ad.index   >= s) & (nav_ad.index   <= e)]
    if b_seg.empty or a_seg.empty:
        continue
    b_ret = b_seg.iloc[-1] / b_seg.iloc[0] - 1
    a_ret = a_seg.iloc[-1] / a_seg.iloc[0] - 1
    b_dd  = ((b_seg - b_seg.cummax()) / b_seg.cummax()).min()
    print(f"{i+1:>3}  {str(s.date())+' ~ '+str(e.date()):>24}  {b_ret:>9.1%}  {a_ret:>9.1%}  {b_dd:>11.1%}")

# ══════════════════════════════════════════════════════
# 诊断 4：北向数据可靠性——2022年前后有效性变化
# ══════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("诊断4：北向资金信号有效性——2022前后分段对比")
print("=" * 65)

# 用滚动IC（信号与未来1月收益的相关性）衡量北向信号质量
bench_rets = close[BENCHMARK].pct_change()
fwd_1m = bench_rets.rolling(21).sum().shift(-21)  # 未来1月沪深300收益
nb_signal = (nb_ma >= 0).astype(int)              # 1=允许买，0=禁买
signal_df = pd.DataFrame({"signal": nb_signal, "fwd_ret": fwd_1m}).dropna()

for period_start, period_end, label in [
    ("2016-01-01", "2022-01-01", "2016-2021（北向开放初期）"),
    ("2022-01-01", "2026-12-31", "2022-2026（北向数据规则变化后）"),
]:
    seg = signal_df[(signal_df.index >= period_start) & (signal_df.index < period_end)]
    if len(seg) < 50:
        print(f"  {label}：数据不足，跳过")
        continue
    ic  = seg["signal"].corr(seg["fwd_ret"])
    hit = (((seg["signal"] == 1) & (seg["fwd_ret"] > 0)) |
           ((seg["signal"] == 0) & (seg["fwd_ret"] < 0))).mean()
    print(f"\n  {label}")
    print(f"  IC={ic:.3f}，方向胜率={hit:.1%}，样本数={len(seg)}")

# ══════════════════════════════════════════════════════
# 可视化：区制分布 + 北向禁买时段 + 净值对比
# ══════════════════════════════════════════════════════
print("\n\n绘制诊断图...")

fig, axes = plt.subplots(4, 1, figsize=(15, 14),
                         gridspec_kw={"height_ratios": [2, 1, 1, 2]})

# (a) 净值对比：基线 vs A+D vs 方向C
bench  = close[BENCHMARK].dropna()
bench_nav = bench / bench.iloc[0] * INIT_CASH
nav_c  = run_backtest(close, scores, rebal_dates, top_n_regime=regime)

ax0 = axes[0]
ax0.plot(nav_base.index,  nav_base  / INIT_CASH, label="基线", color="#9E9E9E", lw=1.8)
ax0.plot(nav_ad.index,    nav_ad    / INIT_CASH, label="方向A+D: LW+北向", color="#2196F3", lw=1.8)
ax0.plot(nav_c.index,     nav_c     / INIT_CASH, label="方向C: 区制", color="#E53935", lw=1.4, alpha=0.8)
ax0.plot(bench_nav.index, bench_nav / INIT_CASH, label="沪深300", color="#FF9800",
         linestyle="--", lw=1.1, alpha=0.6)
ax0.axvline(split_date, color="red", linestyle="--", alpha=0.5, lw=1)
ax0.set_title("净值曲线对比")
ax0.set_ylabel("净值")
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

# (b) 区制时序
ax1 = axes[1]
colors_regime = {1: "#E53935", 2: "#FF9800", 3: "#43A047"}
for r_val, r_color in colors_regime.items():
    mask = (regime == r_val)
    ax1.fill_between(regime.index, 0, 1, where=mask.values,
                     color=r_color, alpha=0.5, transform=ax1.get_xaxis_transform(),
                     label={1:"熊市Top1", 2:"震荡Top2", 3:"牛市Top3"}[r_val])
ax1.plot(pe_pct_s.index, pe_pct_s, color="navy", lw=0.8, alpha=0.7, label="PE分位数")
ax1.axhline(0.80, color="red",  linestyle="--", lw=0.8, alpha=0.6)
ax1.axhline(0.60, color="green", linestyle="--", lw=0.8, alpha=0.6)
ax1.set_title("市场区制（红线=80%分位，绿线=60%分位）")
ax1.set_ylabel("PE分位")
ax1.legend(fontsize=7.5, ncol=5)
ax1.grid(alpha=0.3)

# (c) 北向资金 MA 与禁买信号
ax2 = axes[2]
ax2.plot(nb_ma.index, nb_ma, color="#1A237E", lw=0.9, label=f"北向净买额MA{NORTHBOUND_MA}（亿元）")
ax2.fill_between(nb_ma.index, 0, nb_ma, where=(nb_ma >= 0), alpha=0.3, color="#43A047", label="净流入（允许买）")
ax2.fill_between(nb_ma.index, 0, nb_ma, where=(nb_ma < 0),  alpha=0.3, color="#E53935", label="净流出（禁止买）")
ax2.axhline(0, color="black", lw=0.7)
ax2.set_title("北向资金（沪股通）净买额20日均线")
ax2.set_ylabel("亿元")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# (d) 方向A+D vs 基线回撤
ax3 = axes[3]
dd_base = (nav_base - nav_base.cummax()) / nav_base.cummax() * 100
dd_ad   = (nav_ad   - nav_ad.cummax())   / nav_ad.cummax()   * 100
ax3.fill_between(dd_base.index, dd_base, 0, alpha=0.4, color="#9E9E9E", label=f"基线 MaxDD={dd_base.min():.1f}%")
ax3.fill_between(dd_ad.index,   dd_ad,   0, alpha=0.4, color="#2196F3", label=f"A+D  MaxDD={dd_ad.min():.1f}%")
ax3.set_title("回撤对比：基线 vs 方向A+D")
ax3.set_ylabel("回撤(%)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "etf_rotation_v3_diagnosis.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"诊断图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
