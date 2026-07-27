"""
第二阶段优化验证（2026-07）

测试三个方向：
  1. 标的池去重：删除 159632（中证1000ETF，与512100同指数，重复）
  2. 子行业集中度约束：同组ETF最多2个坑位，顺延到组外次高分
  3. 成交量确认：先做IC分析，IC>=0.05则继续全量回测

基线：风险调整动量 + 拥挤度修正（threshold=0.75, factor=0.2），夏普≈1.005
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
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
BENCHMARK        = "510300.SH"
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21
CORR_WINDOW      = 60
CORR_HIST_WINDOW = 252
CROWD_THRESHOLD  = 0.75
CROWD_FACTOR     = 0.2

# 去重：与512100.SH同指数（中证1000），历史更短，删除
DEDUP_EXCLUDE = {"159632.SZ"}

# 子行业分组（A股中信一级行业对应）
# 宽基不限制，行业ETF按主题分组，组内最多2个坑位
SECTOR_GROUPS = {
    "TMT":   {"515000.SH", "512760.SH", "159995.SZ", "512980.SH",
              "159869.SZ", "513050.SH", "516950.SH"},
    "新能源": {"515330.SH", "515030.SH", "516160.SH", "159629.SZ", "159596.SZ"},
    "医药":   {"512010.SH", "512170.SH", "159992.SZ"},
    "金融地产":{"512800.SH", "512880.SH", "159931.SZ"},
    "消费":   {"159928.SZ", "515700.SH", "159997.SZ", "159801.SZ"},
    "周期":   {"512400.SH", "516670.SH", "515220.SH", "159611.SZ"},
    "军工":   {"512660.SH", "159975.SZ"},
}
# 宽基不限制（无分组约束）

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 工具函数 ──────────────────────────────────────────────

def momentum_score_single(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close: pd.DataFrame) -> pd.DataFrame:
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


def calc_crowding(close: pd.DataFrame) -> pd.DataFrame:
    codes = [c for c in close.columns if c != BENCHMARK]
    ret   = close[codes].pct_change()
    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_WINDOW, len(close.index)):
        ret_win = ret.iloc[i - CORR_WINDOW: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(np.nanmean(corr_arr, axis=1), index=ret_win.columns)
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values
    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_HIST_WINDOW + CORR_WINDOW, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - CORR_HIST_WINDOW: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna()
            c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def apply_crowding(scores: pd.Series, crowding_pct: pd.DataFrame, date) -> pd.Series:
    """对动量得分施加拥挤度惩罚"""
    ds = scores.copy()
    if date in crowding_pct.index:
        dc = crowding_pct.loc[date]
        for code in ds.index:
            if code in dc.index and not pd.isna(dc[code]) and dc[code] > CROWD_THRESHOLD:
                ds[code] *= CROWD_FACTOR
    return ds


def apply_sector_constraint(scores: pd.Series) -> list:
    """
    子行业集中度约束：同组最多2个坑位。
    遍历得分降序排列，每个组内已有2个时跳过，直到选满TOP_N。
    """
    pos_scores = scores[scores > 0].sort_values(ascending=False)
    group_count = {g: 0 for g in SECTOR_GROUPS}
    selected = []
    for code in pos_scores.index:
        if len(selected) >= TOP_N:
            break
        # 判断该ETF属于哪个组
        in_group = None
        for g, members in SECTOR_GROUPS.items():
            if code in members:
                in_group = g
                break
        # 宽基不限制
        if in_group is None or group_count.get(in_group, 0) < 2:
            selected.append(code)
            if in_group:
                group_count[in_group] = group_count.get(in_group, 0) + 1
    return selected


def get_rebalance_dates(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def run_bt(close: pd.DataFrame, scores: pd.DataFrame, rebal_dates: list,
           crowding_pct: pd.DataFrame, use_sector_constraint: bool = False) -> pd.Series:
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

        ds = scores.loc[date].dropna().copy() if date in scores.index else pd.Series(dtype=float)
        ds = apply_crowding(ds, crowding_pct, date)

        if use_sector_constraint:
            tc = apply_sector_constraint(ds)
        else:
            tc = list(ds[ds > 0].nlargest(TOP_N).index)

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
            bp  = price * (1 + SLIPPAGE / 2)
            tv  = pv * weights[code]
            cs  = holdings.get(code, 0)
            diff = tv - cs * price
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
    wins  = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    losses = monthly[monthly < 0].abs().mean() if (monthly < 0).any() else 1
    return {
        "配置": label,
        "年化收益": f"{cagr*100:.1f}%",
        "夏普":     f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%",
        "Calmar":   f"{calmar:.2f}",
        "月胜率":   f"{win_rate:.1%}",
        "_sharpe":  sharpe,
        "_maxdd":   max_dd,
        "_cagr":    cagr,
    }


# ── 成交量IC分析 ──────────────────────────────────────────

def load_amount_matrix() -> pd.DataFrame:
    """加载所有ETF的成交额矩阵（与close_matrix同结构）"""
    daily_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "daily"
    dfs = {}
    for f in daily_dir.glob("*.parquet"):
        code = f.stem
        df = pd.read_parquet(f, columns=["trade_date", "amount"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        dfs[code] = df["amount"]
    if not dfs:
        return pd.DataFrame()
    result = pd.DataFrame(dfs)
    # 截断末尾NaN行（与close_matrix保持一致）
    min_valid_rows = result.shape[1] // 2
    valid_mask = result.notna().sum(axis=1) >= min_valid_rows
    last_valid = result[valid_mask].index[-1]
    return result[result.index <= last_valid]


def calc_vol_ic(close: pd.DataFrame, amount: pd.DataFrame, rebal_dates: list) -> pd.DataFrame:
    """
    计算成交量确认因子的IC（信息系数）。
    vol_ratio = MA5(amount) / MA20(amount)，月度信号日计算。
    IC = spearman相关(vol_ratio排名, 下月收益率排名)
    """
    ic_rows = []
    codes = [c for c in close.columns if c in amount.columns and c != BENCHMARK]

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]
        # 计算vol_ratio
        vol_ratios = {}
        for code in codes:
            amt = amount[code].dropna()
            idx = amt.index.get_indexer([date], method="ffill")[0]
            if idx < 20:
                continue
            ma5  = amt.iloc[max(0, idx - 5): idx].mean()
            ma20 = amt.iloc[max(0, idx - 20): idx].mean()
            if ma20 > 0:
                vol_ratios[code] = ma5 / ma20
        if len(vol_ratios) < 5:
            continue

        # 计算下月收益率
        fwd_rets = {}
        for code in vol_ratios:
            p0 = close[code].asof(date)
            p1 = close[code].asof(next_date)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_rets[code] = p1 / p0 - 1

        common = set(vol_ratios) & set(fwd_rets)
        if len(common) < 5:
            continue

        vr = pd.Series({c: vol_ratios[c] for c in common})
        fr = pd.Series({c: fwd_rets[c]  for c in common})
        ic = vr.rank().corr(fr.rank(), method="spearman")
        ic_rows.append({"date": date, "IC": ic, "n": len(common)})

    return pd.DataFrame(ic_rows).set_index("date") if ic_rows else pd.DataFrame()


# ── 主流程 ────────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]

# 基线（含全部45只）
valid_all = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close_all = close[valid_all]

# 去重版本（排除159632）
valid_dedup = [c for c in valid_all if c not in DEDUP_EXCLUDE]
close_dedup = close[valid_dedup]

print(f"基线标的数：{len(valid_all)}，去重后：{len(valid_dedup)}")
print(f"日期范围：{close.index[0].date()} ~ {close.index[-1].date()}")

# ── 计算共用的动量/拥挤度（基线） ────────────────────────
print("计算动量得分（基线）...")
scores_all   = calc_all_scores(close_all)
print("计算拥挤度（基线）...")
crowding_all = calc_crowding(close_all)
rebal_all    = [d for d in get_rebalance_dates(close_all.index) if d >= pd.Timestamp(START_DATE)]

print("计算动量得分（去重版）...")
scores_dedup   = calc_all_scores(close_dedup)
print("计算拥挤度（去重版）...")
crowding_dedup = calc_crowding(close_dedup)

n_days     = len(close_all)
split_idx  = int(n_days * IS_RATIO)
split_date = close_all.index[split_idx]

print(f"IS/OOS分割：{close_all.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close_all.index[-1].date()}")

# ── 回测：4种配置 ─────────────────────────────────────────
print("\n运行回测...")

configs = [
    ("基线（拥挤度修正）",              close_all,  scores_all,   crowding_all,   False),
    ("去重（删159632）",               close_dedup, scores_dedup, crowding_dedup, False),
    ("子行业约束（基线池）",             close_all,  scores_all,   crowding_all,   True),
    ("子行业约束+去重",                 close_dedup, scores_dedup, crowding_dedup, True),
]

rebal_dedup = [d for d in get_rebalance_dates(close_dedup.index) if d >= pd.Timestamp(START_DATE)]

navs_full = {}
navs_is   = {}
navs_oos  = {}

for label, cl, sc, cp, use_sec in configs:
    rebal = rebal_all if cl is close_all else rebal_dedup
    cl_is  = cl[cl.index <  split_date]
    cl_oos = cl[cl.index >= split_date]
    sc_is  = sc[sc.index <  split_date]
    sc_oos = sc[sc.index >= split_date]
    cp_is  = cp[cp.index <  split_date]
    cp_oos = cp[cp.index >= split_date]
    rb_is  = [d for d in rebal if d <  split_date]
    rb_oos = [d for d in rebal if d >= split_date]

    navs_full[label] = run_bt(cl,     sc,     rebal,  cp,     use_sec)
    navs_is[label]   = run_bt(cl_is,  sc_is,  rb_is,  cp_is,  use_sec)
    navs_oos[label]  = run_bt(cl_oos, sc_oos, rb_oos, cp_oos, use_sec)
    s = calc_stats(navs_full[label])
    print(f"  {label:<28} 全样本夏普={s['_sharpe']:.3f}")

# ── 结果输出 ──────────────────────────────────────────────
print("\n" + "=" * 75)
print("全样本全量指标（2016-2026）")
print("=" * 75)
rows = [calc_stats(navs_full[l], l) for l, *_ in configs]
df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
print(df_out.set_index("配置").to_string())

base_sharpe = calc_stats(navs_full["基线（拥挤度修正）"])["_sharpe"]

print("\n" + "=" * 75)
print("IS / OOS 验证")
print("=" * 75)
for label, *_ in configs:
    si = calc_stats(navs_is[label])
    so = calc_stats(navs_oos[label])
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    vs_base = calc_stats(navs_full[label])["_sharpe"] - base_sharpe
    status = "通过" if decay >= 0.5 else "警告:过拟合"
    print(f"\n{label}")
    print(f"  IS ：夏普={si['_sharpe']:.3f}  年化={si['年化收益']}  回撤={si['最大回撤']}")
    print(f"  OOS：夏普={so['_sharpe']:.3f}  年化={so['年化收益']}  回撤={so['最大回撤']}")
    print(f"  OOS/IS={decay:.2f}  [{status}]  全样本vs基线：{vs_base:+.3f}")

# ── 逐年对比 ──────────────────────────────────────────────
print("\n" + "=" * 75)
print("逐年收益对比")
print("=" * 75)
show_labels = [l for l, *_ in configs]
print(f"{'年份':<6}", end="")
for l in show_labels:
    print(f"  {l[:16]:>16}", end="")
print()
for yr in sorted(set(navs_full[show_labels[0]].index.year)):
    s = pd.Timestamp(f"{yr}-01-01")
    e = pd.Timestamp(f"{yr}-12-31")
    print(f"{yr:<6}", end="")
    base_ret = None
    for label in show_labels:
        nav = navs_full[label]
        seg = nav[(nav.index >= s) & (nav.index <= e)]
        if seg.empty:
            print(f"  {'—':>16}", end="")
        else:
            ret = seg.iloc[-1] / seg.iloc[0] - 1
            if base_ret is None:
                base_ret = ret
                marker = ""
            else:
                marker = " ↑" if ret > base_ret + 0.005 else (" ↓" if ret < base_ret - 0.005 else "")
            print(f"  {ret*100:>14.1f}%{marker}", end="")
    print()

# ── 成交量IC分析 ──────────────────────────────────────────
print("\n" + "=" * 75)
print("成交量确认因子 IC 分析")
print("=" * 75)
print("加载成交额数据...")
amount = load_amount_matrix()
if not amount.empty:
    amount = amount[amount.index >= START_DATE]
    ic_df = calc_vol_ic(close_all, amount, rebal_all)
    if not ic_df.empty:
        mean_ic  = ic_df["IC"].mean()
        ic_ir    = ic_df["IC"].mean() / ic_df["IC"].std() if ic_df["IC"].std() > 0 else 0
        pos_rate = (ic_df["IC"] > 0).mean()
        print(f"月均IC    = {mean_ic:+.4f}")
        print(f"IC IR     = {ic_ir:+.4f}")
        print(f"IC>0占比  = {pos_rate:.1%}")
        print(f"测试月数  = {len(ic_df)}")
        if abs(mean_ic) >= 0.05:
            print("→ IC显著（≥0.05），建议继续全量回测")
        else:
            print("→ IC不显著（<0.05），成交量确认信号无效，放弃")
    else:
        print("IC计算失败（数据不足）")
else:
    print("成交额数据加载失败")

# ── 可视化 ────────────────────────────────────────────────
out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)

colors = {
    "基线（拥挤度修正）":   "#9E9E9E",
    "去重（删159632）":     "#1565C0",
    "子行业约束（基线池）":  "#E53935",
    "子行业约束+去重":      "#43A047",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax1 = axes[0]
for label, *_ in configs:
    nav = navs_full[label]
    lw  = 2.0 if "基线" in label else 1.6
    ls  = "--" if "基线" in label else "-"
    ax1.plot(nav.index, nav / INIT_CASH, label=label,
             color=colors.get(label, "gray"), linewidth=lw, linestyle=ls)
bench = close_all[BENCHMARK].dropna()
bench = bench / bench.iloc[0] * INIT_CASH
ax1.plot(bench.index, bench / INIT_CASH, color="#FF9800", linestyle=":", lw=1.0, alpha=0.6, label="沪深300")
ax1.axvline(split_date, color="red", linestyle="--", alpha=0.4, lw=1)
ax1.set_title("第二阶段优化验证：去重 + 子行业集中度约束（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2 = axes[1]
for label, *_ in configs:
    nav = navs_full[label]
    dd  = (nav - nav.cummax()) / nav.cummax() * 100
    lw  = 1.8 if "基线" in label else 1.4
    ax2.plot(dd.index, dd, label=f"{label}  MaxDD={dd.min():.1f}%",
             color=colors.get(label, "gray"), linewidth=lw)
ax2.set_ylabel("回撤(%)")
ax2.set_title("回撤对比")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "etf_rotation_p2_validate.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
