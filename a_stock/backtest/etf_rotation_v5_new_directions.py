"""
ETF 轮动新方向验证（2026-07）

测试5个未经验证的方向，基线为当前最优配置（风险调整动量 + 拥挤度修正0.75/0.2）：

1. Shrinkage（James-Stein 收缩）：动量得分向截面均值收缩，降低小样本估计噪声
2. 惰性调仓（Lazy Rebalance）：持仓标的排名下滑幅度未超过阈值时不换仓，降低换手成本
3. 拥挤度衰减加权（EWM Crowding）：拥挤度相关系数用指数衰减加权，而非等权
4. 信号平滑（Score EMA）：对月度动量得分做跨期 EMA 平滑，降低单月噪声
5. 换手率统计：仅统计基线策略的年均换手率，为判断惰性调仓收益空间提供依据

输出：各方向 vs 基线全量指标对比，及换手率统计报告
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
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
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
CROWD_THRESHOLD  = 0.75   # 基线拥挤度参数
CROWD_FACTOR     = 0.20

# Shrinkage 参数
SHRINK_ALPHA     = 0.5    # 0=不收缩，1=全部用截面均值；测试0.3/0.5/0.7

# 惰性调仓参数
LAZY_MIN_RANK_DROP = 3    # 持仓标的排名下滑少于此名次时不换仓；测试2/3/5

# 信号平滑参数
SCORE_EMA_SPAN   = 3      # EMA span（月数）；测试2/3/4

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 核心计算函数 ─────────────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    """OLS斜率 × R²（年化）"""
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """计算每个交易日各标的的风险调整动量得分"""
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score(series.iloc[i - MOMENTUM_WINDOW: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol  = rets.std() * np.sqrt(252)
                raw  = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close_matrix.index)


def calc_crowding_equal(close: pd.DataFrame) -> pd.DataFrame:
    """等权相关系数拥挤度（基线方式）"""
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
            h = hist[code].dropna(); c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def calc_crowding_ewm(close: pd.DataFrame, halflife: int = 20) -> pd.DataFrame:
    """指数衰减加权相关系数拥挤度（方向3：拥挤度衰减加权）"""
    codes = [c for c in close.columns if c != BENCHMARK]
    ret   = close[codes].pct_change()
    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    # 预计算每日的指数衰减权重
    weights = np.array([0.5 ** (k / halflife) for k in range(CORR_WINDOW - 1, -1, -1)])
    weights = weights / weights.sum()
    for i in range(CORR_WINDOW, len(close.index)):
        ret_win = ret.iloc[i - CORR_WINDOW: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        # 加权协方差 -> 加权相关
        vals = ret_win.values  # shape (CORR_WINDOW, n_codes)
        w_mean = (vals * weights[:, None]).sum(axis=0)
        demeaned = vals - w_mean[None, :]
        wcov = (demeaned * weights[:, None]).T @ demeaned  # (n, n)
        std = np.sqrt(np.diag(wcov))
        with np.errstate(invalid="ignore", divide="ignore"):
            wcorr = wcov / np.outer(std, std)
        np.fill_diagonal(wcorr, np.nan)
        avg_corr = pd.Series(np.nanmean(wcorr, axis=1), index=ret_win.columns)
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values
    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_HIST_WINDOW + CORR_WINDOW, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - CORR_HIST_WINDOW: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna(); c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def apply_shrinkage(scores_at_date: pd.Series, alpha: float) -> pd.Series:
    """
    James-Stein 向截面均值收缩：
    adj_score = (1 - alpha) * raw_score + alpha * cross_section_mean
    alpha=0 → 不收缩；alpha=1 → 全部用截面均值
    """
    valid = scores_at_date.dropna()
    if valid.empty:
        return scores_at_date
    cross_mean = valid.mean()
    return (1 - alpha) * scores_at_date + alpha * cross_mean


def get_rebalance_dates(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 回测引擎 ─────────────────────────────────────────────

def run_bt(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    crowding_pct: pd.DataFrame,
    *,
    mode: str = "baseline",   # baseline / shrink / lazy / score_ema
    shrink_alpha: float = 0.5,
    lazy_min_rank_drop: int = 3,
    score_ema_span: int = 3,
    crowd_threshold: float = CROWD_THRESHOLD,
    crowd_factor: float = CROWD_FACTOR,
    crowding_pct_ewm: pd.DataFrame = None,  # 方向3专用
    track_turnover: bool = False,
) -> tuple[pd.Series, dict]:
    """
    统一回测引擎，通过 mode 控制变体：
      baseline   : 基线（拥挤度等权）
      shrink     : 动量得分向截面均值收缩
      lazy       : 惰性调仓（排名下滑 < lazy_min_rank_drop 不换）
      score_ema  : 对月度得分做 EMA 平滑
      ewm_crowd  : 拥挤度改用指数衰减加权
    track_turnover: True 时额外统计换手率
    返回 (nav_series, meta_dict)，meta_dict 含换手率等辅助信息
    """
    cash = INIT_CASH
    holdings: dict[str, float] = {}   # code -> shares
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set  = set(rebal_dates)

    # 信号平滑：维护每个标的的 EMA 得分
    ema_scores: dict[str, float] = {}
    ema_alpha = 2.0 / (score_ema_span + 1)

    # 换手率追踪
    monthly_turnover: list[float] = []
    total_value_for_turnover: list[float] = []

    for date in close.index:
        # 计算当日组合净值
        pv = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                pv += shares * close.loc[date, code]
        nav_series[date] = pv

        if date not in rebal_set:
            continue

        # ── 计算调仓信号 ──────────────────────────────────
        ds = scores.loc[date].dropna().copy()

        # 方向4：信号 EMA 平滑
        if mode == "score_ema":
            for code in ds.index:
                prev = ema_scores.get(code)
                if prev is None:
                    ema_scores[code] = ds[code]
                else:
                    ema_scores[code] = ema_alpha * ds[code] + (1 - ema_alpha) * prev
            ds = pd.Series(ema_scores).reindex(ds.index).dropna()

        # 方向1：Shrinkage
        if mode == "shrink":
            ds = apply_shrinkage(ds, shrink_alpha)

        # 拥挤度修正（基线 / shrink / lazy / score_ema 都用等权拥挤度）
        cp = crowding_pct_ewm if mode == "ewm_crowd" else crowding_pct
        if cp is not None and date in cp.index:
            dc = cp.loc[date]
            for code in ds.index:
                if code in dc.index and not pd.isna(dc[code]) and dc[code] > crowd_threshold:
                    ds[code] *= crowd_factor

        # 候选持仓
        pos_ds  = ds[ds > 0]
        all_ranked = list(pos_ds.sort_values(ascending=False).index)
        target = all_ranked[:TOP_N]

        # 方向2：惰性调仓 ——
        # 对当前每个持仓标的，若其在 all_ranked 中的排名下滑幅度 < lazy_min_rank_drop，
        # 则保留该标的，用排名更靠后的替代者补位。
        if mode == "lazy" and holdings:
            sticky = []
            for code in list(holdings.keys()):
                if code not in pos_ds.index:
                    continue  # 无正动量，不保留
                new_rank = all_ranked.index(code) if code in all_ranked else 999
                # 若原来在 TOP_N 内 or 排名下滑 < lazy_min_rank_drop，保留
                if new_rank < TOP_N + lazy_min_rank_drop:
                    sticky.append(code)
            # 用 sticky 优先填坑，不足 TOP_N 再从 all_ranked 顺序补
            merged = sticky[:]
            for code in all_ranked:
                if code not in merged:
                    merged.append(code)
            target = merged[:TOP_N]

        # ── 执行调仓 ──────────────────────────────────────
        trade_value = 0.0  # 记录本次换手金额

        # 卖出不在 target 的持仓
        for code in list(holdings.keys()):
            if code not in target:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    sell_val = holdings[code] * price
                    cash += sell_val * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    if track_turnover:
                        trade_value += sell_val
                del holdings[code]

        if not target:
            if track_turnover and pv > 0:
                monthly_turnover.append(trade_value / pv)
            continue

        n = len(target)
        weights = {c: 1.0 / n for c in target}

        for code in target:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            bp  = price * (1 + SLIPPAGE / 2)
            tv  = pv * weights[code]
            cs  = holdings.get(code, 0)
            cv  = cs * price
            diff = tv - cv
            if diff > bp * 100:
                bs = int(diff / bp / 100) * 100
                if bs > 0:
                    cost = bs * bp * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = cs + bs
                        if track_turnover:
                            trade_value += bs * bp
            elif diff < -price * 100:
                ss = int(-diff / price / 100) * 100
                if ss > 0 and cs >= ss:
                    cash += ss * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = cs - ss
                    if track_turnover:
                        trade_value += ss * price

        if track_turnover and pv > 0:
            monthly_turnover.append(trade_value / pv)
            total_value_for_turnover.append(pv)

    meta = {}
    if track_turnover and monthly_turnover:
        avg_monthly = np.mean(monthly_turnover)
        meta["avg_monthly_turnover"] = avg_monthly
        meta["annual_turnover"]      = avg_monthly * 12
        meta["n_rebal"]              = len(monthly_turnover)

    return nav_series.dropna(), meta


def calc_full_stats(nav: pd.Series, label: str = "") -> dict:
    rets   = nav.pct_change().dropna()
    years  = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr   = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino  = cagr / downside if downside > 0 else 0
    monthly  = nav.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    wins     = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    losses   = monthly[monthly < 0].abs().mean() if (monthly < 0).any() else 1
    pnl_ratio = wins / losses if losses > 0 else 0
    return {
        "标的":      label,
        "年化收益":  f"{cagr*100:.1f}%",
        "夏普":      f"{sharpe:.3f}",
        "最大回撤":  f"{max_dd*100:.1f}%",
        "Calmar":    f"{calmar:.2f}",
        "Sortino":   f"{sortino:.2f}",
        "月胜率":    f"{win_rate:.1%}",
        "盈亏比":    f"{pnl_ratio:.2f}",
        "_sharpe":   sharpe,
        "_maxdd":    max_dd,
        "_cagr":     cagr,
    }


# ── 加载数据 ──────────────────────────────────────────────

print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
scores = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

print("计算等权拥挤度（基线方式）...")
crowding_eq = calc_crowding_equal(close)

print("计算指数衰减拥挤度（方向3）...")
crowding_ewm = calc_crowding_ewm(close, halflife=20)

n_days     = len(close)
split_idx  = int(n_days * IS_RATIO)
split_date = close.index[split_idx]

# 预切 IS/OOS 数据
close_is  = close[close.index <  split_date]
close_oos = close[close.index >= split_date]
rebal_is  = [d for d in rebal_dates if d <  split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
sc_is  = scores[scores.index <  split_date]
sc_oos = scores[scores.index >= split_date]
cp_is  = crowding_eq[crowding_eq.index <  split_date]
cp_oos = crowding_eq[crowding_eq.index >= split_date]
ewm_is  = crowding_ewm[crowding_ewm.index <  split_date]
ewm_oos = crowding_ewm[crowding_ewm.index >= split_date]

bench_nav = close[BENCHMARK].dropna()
bench_nav = bench_nav / bench_nav.iloc[0] * INIT_CASH

# ── 运行所有变体 ─────────────────────────────────────────

VARIANTS = [
    # (label, mode, kwargs)
    ("基线",                   "baseline", {}),
    (f"方向1 shrink={SHRINK_ALPHA}",  "shrink",  {"shrink_alpha": SHRINK_ALPHA}),
    ("方向1 shrink=0.3",       "shrink",  {"shrink_alpha": 0.3}),
    ("方向1 shrink=0.7",       "shrink",  {"shrink_alpha": 0.7}),
    (f"方向2 lazy_rank={LAZY_MIN_RANK_DROP}", "lazy", {"lazy_min_rank_drop": LAZY_MIN_RANK_DROP}),
    ("方向2 lazy_rank=2",      "lazy",    {"lazy_min_rank_drop": 2}),
    ("方向2 lazy_rank=5",      "lazy",    {"lazy_min_rank_drop": 5}),
    ("方向3 ewm_crowd",        "ewm_crowd", {}),
    (f"方向4 score_ema={SCORE_EMA_SPAN}", "score_ema", {"score_ema_span": SCORE_EMA_SPAN}),
    ("方向4 score_ema=2",      "score_ema", {"score_ema_span": 2}),
    ("方向4 score_ema=4",      "score_ema", {"score_ema_span": 4}),
]

print("\n运行回测变体...")
navs_full = {}
navs_is   = {}
navs_oos  = {}
metas     = {}

for label, mode, kwargs in VARIANTS:
    use_ewm = (mode == "ewm_crowd")
    # ewm_crowd 模式：等权拥挤度传 None，ewm 拥挤度通过 crowding_pct_ewm 传入
    cp_full_eq = None if use_ewm else crowding_eq
    cp_i_eq    = None if use_ewm else cp_is
    cp_o_eq    = None if use_ewm else cp_oos

    nav_f, meta = run_bt(close,     scores, rebal_dates, cp_full_eq, mode=mode,
                         track_turnover=(label == "基线"), **kwargs,
                         crowding_pct_ewm=crowding_ewm if use_ewm else None)
    nav_i, _    = run_bt(close_is,  sc_is,  rebal_is,   cp_i_eq,    mode=mode, **kwargs,
                         crowding_pct_ewm=ewm_is if use_ewm else None)
    nav_o, _    = run_bt(close_oos, sc_oos, rebal_oos,  cp_o_eq,    mode=mode, **kwargs,
                         crowding_pct_ewm=ewm_oos if use_ewm else None)
    navs_full[label] = nav_f
    navs_is[label]   = nav_i
    navs_oos[label]  = nav_o
    metas[label]     = meta
    s = calc_full_stats(nav_f)
    print(f"  {label:<30} 全样本夏普={s['_sharpe']:.3f}")

# ── 换手率统计报告 ───────────────────────────────────────

print("\n" + "=" * 80)
print("方向5：基线换手率统计")
print("=" * 80)
# 重新跑一次基线，开启换手率追踪
nav_to, meta_to = run_bt(close, scores, rebal_dates, crowding_eq,
                          mode="baseline", track_turnover=True)
m = meta_to
if m:
    print(f"调仓次数：{m['n_rebal']}")
    print(f"月均换手率：{m['avg_monthly_turnover']*100:.1f}%")
    print(f"年化换手率：{m['annual_turnover']*100:.1f}%")
    print(f"年化单边交易成本估算：{m['annual_turnover']*100 * 0.0164:.2f}%  (按0.164%/回合)")
    print(f"注：ETF 无印花税，单次完整回合成本约0.164%（佣金+滑点）")

# ── 全量指标对比 ─────────────────────────────────────────

print("\n" + "=" * 80)
print("全样本全量指标对比（2016-2026）")
print("=" * 80)
rows_full = [calc_full_stats(navs_full[l], l) for l, *_ in VARIANTS]
df_full = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in rows_full]).set_index("标的")
print(df_full.to_string())

print(f"\nIS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")
print("\nIS/OOS 验证（夏普）")
print("=" * 80)
is_oos_rows = []
for label, *_ in VARIANTS:
    si = calc_full_stats(navs_is[label])
    so = calc_full_stats(navs_oos[label])
    decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告:可能过拟合"
    vs_base_full = navs_full[label].iloc[-1] / navs_full["基线"].iloc[-1] - 1
    is_oos_rows.append({
        "配置": label,
        "IS夏普":  f"{si['_sharpe']:.3f}",
        "OOS夏普": f"{so['_sharpe']:.3f}",
        "OOS/IS":  f"{decay:.2f}",
        "全样本夏普": f"{calc_full_stats(navs_full[label])['_sharpe']:.3f}",
        "vs基线净值": f"{vs_base_full*100:+.1f}%",
        "状态":    status,
    })
print(pd.DataFrame(is_oos_rows).set_index("配置").to_string())

# ── 结论汇总 ─────────────────────────────────────────────

base_sharpe = calc_full_stats(navs_full["基线"])["_sharpe"]
print("\n" + "=" * 80)
print("结论汇总（vs 基线夏普 {:.3f}）".format(base_sharpe))
print("=" * 80)
for row in rows_full:
    label = row["标的"]
    if label == "基线":
        continue
    delta = row["_sharpe"] - base_sharpe
    tag = "有效" if delta > 0.02 else ("中性" if delta > -0.02 else "有害")
    print(f"  {label:<35}  Δ夏普={delta:+.3f}  [{tag}]")

# ── 可视化 ───────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

# 每个方向单独出一张图（基线 + 该方向各参数）
directions = {
    "方向1_shrinkage":  [l for l, *_ in VARIANTS if "方向1" in l or l == "基线"],
    "方向2_lazy":       [l for l, *_ in VARIANTS if "方向2" in l or l == "基线"],
    "方向3_ewm_crowd":  [l for l, *_ in VARIANTS if "方向3" in l or l == "基线"],
    "方向4_score_ema":  [l for l, *_ in VARIANTS if "方向4" in l or l == "基线"],
}

colors_map = {
    "基线":       "#9E9E9E",
    "方向1 shrink=0.3": "#1565C0",
    f"方向1 shrink={SHRINK_ALPHA}": "#E53935",
    "方向1 shrink=0.7": "#FF7043",
    "方向2 lazy_rank=2": "#1565C0",
    f"方向2 lazy_rank={LAZY_MIN_RANK_DROP}": "#E53935",
    "方向2 lazy_rank=5": "#FF7043",
    "方向3 ewm_crowd":   "#E53935",
    "方向4 score_ema=2": "#1565C0",
    f"方向4 score_ema={SCORE_EMA_SPAN}": "#E53935",
    "方向4 score_ema=4": "#FF7043",
}

for dir_name, labels in directions.items():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [3, 1.5]})
    ax1, ax2 = axes

    for lbl in labels:
        nav  = navs_full[lbl]
        lw   = 2.0 if lbl == "基线" else 1.5
        ls   = "--" if lbl == "基线" else "-"
        color = colors_map.get(lbl, "gray")
        s = calc_full_stats(nav)
        ax1.plot(nav.index, nav / INIT_CASH, label=f"{lbl[:30]}  Sharpe={s['_sharpe']:.3f}",
                 lw=lw, ls=ls, color=color)

    ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800",
             ls=":", lw=0.9, alpha=0.6, label="沪深300")
    ax1.axvline(split_date, color="red", ls="--", alpha=0.4, lw=1.0, label="IS/OOS分割")
    ax1.set_title(f"ETF轮动新方向验证 — {dir_name}（2016-2026）")
    ax1.set_ylabel("净值")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="gray", ls="--", alpha=0.3)

    for lbl in labels:
        nav = navs_full[lbl]
        dd  = (nav - nav.cummax()) / nav.cummax() * 100
        lw  = 1.8 if lbl == "基线" else 1.3
        ax2.plot(dd.index, dd, lw=lw, ls="--" if lbl == "基线" else "-",
                 color=colors_map.get(lbl, "gray"),
                 label=f"{lbl[:20]} MaxDD={dd.min():.1f}%")

    ax2.set_ylabel("回撤(%)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("回撤对比")
    plt.tight_layout()
    fig_path = out_dir / f"etf_rotation_v5_{dir_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"图已保存：{fig_path}")

print("\n完成。")
