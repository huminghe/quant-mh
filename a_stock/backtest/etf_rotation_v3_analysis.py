"""
ETF 轮动新优化方向测试（v3，2026-07）
基于数据可用性验证，测试以下三个方向：

  方向A: Ledoit-Wolf 最小方差权重
         用协方差收缩估计替代等权分配，降低持仓间相关性带来的风险集中
  方向C: 市场状态区制切换
         用沪深300 PE分位数 + 近期波动率定义牛/熊/震荡三种状态，
         高估值+高波动（熊市/震荡）时降持仓数（从Top3降到Top2/1）
  方向D: 北向资金辅助过滤
         北向净买额20日均线为负时，禁止新增仓位（防止外资撤离期追涨）

输出：
  - 全样本对比表（含基线）
  - IS/OOS 验证（所有配置，含样本外夏普比较）
  - 净值曲线图
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
from fetch_data import load_close_matrix, init_pro

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
BENCHMARK        = "510300.SH"
BENCHMARK_INDEX  = "000300.SH"   # 沪深300指数代码（用于拉 PE 数据）
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21

# 方向C 参数
PE_WINDOW = 252        # PE历史分位数计算窗口（1年）
VOL_WINDOW_C = 20      # 波动率窗口

# 方向D 参数
NORTHBOUND_MA = 20     # 北向资金净买额均线周期

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 辅助数据加载 ───────────────────────────────────────────

def load_pe_data() -> pd.Series:
    """从 tushare 拉取沪深300历史 PE_TTM"""
    print("  加载沪深300 PE数据...")
    pro = init_pro()
    df = pro.index_dailybasic(
        ts_code=BENCHMARK_INDEX,
        start_date="20150101",
        end_date="20261231",
        fields="trade_date,pe_ttm",
    )
    if df is None or df.empty:
        raise RuntimeError("index_dailybasic 返回空数据")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df["pe_ttm"]


def load_northbound_data() -> pd.Series:
    """
    从 akshare 拉取沪股通历史净买额。
    返回按日期索引的净买额 Series（亿元），数据从2014年起。
    """
    print("  加载北向资金（沪股通）历史数据...")
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol="沪股通")
        if df is None or df.empty:
            print("  [警告] 北向资金数据为空，方向D将跳过")
            return pd.Series(dtype=float)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").set_index("日期")
        net_buy = df["当日成交净买额"].astype(float)
        print(f"  北向资金数据范围：{net_buy.index[0].date()} ~ {net_buy.index[-1].date()}")
        return net_buy
    except Exception as e:
        print(f"  [警告] 北向资金数据加载失败：{e}，方向D将跳过")
        return pd.Series(dtype=float)


# ── 市场状态区制（方向C）───────────────────────────────────

def build_regime(
    close: pd.DataFrame,
    pe_series: pd.Series,
    pe_window: int = PE_WINDOW,
    vol_window: int = VOL_WINDOW_C,
) -> pd.Series:
    """
    定义市场区制：
    - 牛市（regime=3）：PE分位数 < 0.6 AND 近期波动率 < 历史60%分位数
    - 震荡市（regime=2）：其余
    - 熊市（regime=1）：PE分位数 > 0.8 OR 近期波动率 > 历史80%分位数
    返回 pd.Series，values=1/2/3，index=date，对应持仓数 Top1/2/3
    """
    # 对齐 PE 到交易日
    pe = pe_series.reindex(close.index).ffill()

    # 沪深300日收益率 → 近期波动率
    if BENCHMARK in close.columns:
        bench_ret = close[BENCHMARK].pct_change()
    else:
        bench_ret = pd.Series(dtype=float, index=close.index)

    vol_20 = bench_ret.rolling(vol_window).std() * np.sqrt(252)

    regime = pd.Series(3, index=close.index, dtype=int)  # 默认牛市

    for i in range(pe_window, len(close.index)):
        date = close.index[i]
        curr_pe = pe.iloc[i]
        curr_vol = vol_20.iloc[i]
        if pd.isna(curr_pe) or pd.isna(curr_vol):
            continue

        # 历史 PE 分位数
        hist_pe = pe.iloc[i - pe_window: i].dropna()
        pe_pct = (hist_pe < curr_pe).mean() if len(hist_pe) >= 50 else 0.5

        # 历史波动率分位数
        hist_vol = vol_20.iloc[i - pe_window: i].dropna()
        vol_pct = (hist_vol < curr_vol).mean() if len(hist_vol) >= 50 else 0.5

        # 区制判断
        is_bear = pe_pct > 0.80 or vol_pct > 0.80
        is_bull = pe_pct < 0.60 and vol_pct < 0.60

        if is_bear:
            regime[date] = 1   # 熊市：Top1
        elif is_bull:
            regime[date] = 3   # 牛市：Top3
        else:
            regime[date] = 2   # 震荡：Top2

    return regime


# ── 北向资金过滤（方向D）──────────────────────────────────

def build_northbound_filter(
    close: pd.DataFrame,
    nb_series: pd.Series,
    ma_window: int = NORTHBOUND_MA,
) -> pd.Series:
    """
    计算北向净买额 MA，返回布尔 Series：
    True = 允许买入（北向净流入趋势）
    False = 禁止新增仓位（北向撤离）
    """
    if nb_series.empty:
        return pd.Series(True, index=close.index)

    nb_ma = nb_series.rolling(ma_window).mean()
    # 对齐到交易日
    nb_ma_aligned = nb_ma.reindex(close.index).ffill()
    allow_buy = nb_ma_aligned >= 0
    allow_buy = allow_buy.fillna(True)  # 无数据时不过滤
    return allow_buy


# ── 信号计算（与 v2 一致）─────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    """OLS 斜率×R² 年化动量得分"""
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(
    close_matrix: pd.DataFrame,
    window: int = MOMENTUM_WINDOW,
    risk_adj: bool = True,
    risk_vol_window: int = RISK_VOL_WINDOW,
) -> pd.DataFrame:
    """计算所有标的的风险调整动量得分矩阵"""
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            pw = series.iloc[i - window: i]
            raw = momentum_score(pw)
            if risk_adj and i >= risk_vol_window:
                rets = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            score_series.iloc[i] = raw
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 回测主逻辑 ─────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    # 方向A
    use_ledoit_wolf: bool = False,
    lw_min_history: int = 60,      # 协方差估计最少需要的历史天数
    # 方向C
    top_n_regime: pd.Series = None,  # 预计算的区制序列（1/2/3）
    # 方向D
    allow_buy_filter: pd.Series = None,  # 北向资金允许买入过滤
) -> pd.Series:
    """
    月度轮换回测，支持三个新优化方向的开关。
    """
    # Ledoit-Wolf 延迟导入（可选依赖）
    if use_ledoit_wolf:
        from sklearn.covariance import LedoitWolf

    cash = init_cash
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    for date in close.index:
        # 计算净值
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        if date not in rebal_set:
            continue

        # 当日有效得分
        day_scores = scores.loc[date].dropna()

        # 方向C：动态持仓数
        effective_n = int(top_n_regime[date]) if (
            top_n_regime is not None and date in top_n_regime.index
        ) else top_n

        # 初筛：正动量候选
        pos_scores = day_scores[day_scores > 0].nlargest(effective_n * 3)
        candidates = list(pos_scores.index)
        target_codes = candidates[:effective_n]

        # 卖出不在目标中的持仓
        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        # 方向D：北向资金过滤——禁止新增仓位时，维持现有持仓，不买入新标的
        if allow_buy_filter is not None:
            nb_allow = bool(allow_buy_filter.get(date, True))
        else:
            nb_allow = True

        # 方向A：Ledoit-Wolf 权重 vs 等权
        n = len(target_codes)
        if use_ledoit_wolf and n >= 2 and nb_allow:
            date_loc = close.index.get_loc(date)
            hist_start = max(0, date_loc - 252)  # 最多用1年历史
            ret_hist = close[target_codes].iloc[hist_start:date_loc].pct_change().dropna()
            if len(ret_hist) >= lw_min_history:
                try:
                    lw = LedoitWolf().fit(ret_hist.values)
                    cov = lw.covariance_
                    # 最小方差权重（无约束解析解）
                    ones = np.ones(n)
                    inv_cov = np.linalg.pinv(cov)
                    raw_w = inv_cov @ ones
                    # 限制权重在 [0.05, 0.70] 区间，防止极端集中
                    raw_w = np.clip(raw_w, 0.05, None)
                    raw_w = np.clip(raw_w, None, 0.70 * raw_w.sum())
                    w_arr = raw_w / raw_w.sum()
                    weights = {code: float(w_arr[i]) for i, code in enumerate(target_codes)}
                except Exception:
                    weights = {c: 1.0 / n for c in target_codes}
            else:
                weights = {c: 1.0 / n for c in target_codes}
        else:
            weights = {c: 1.0 / n for c in target_codes}

        # 买入目标持仓
        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue

            # 方向D：北向资金禁买时，跳过新标的建仓（已有持仓照常持有）
            if not nb_allow and code not in holdings:
                continue

            buy_price = price * (1 + SLIPPAGE / 2)
            target_value = port_value * weights[code]
            current_shares = holdings.get(code, 0)
            current_value = current_shares * price
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
                    cash += sell_shares * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


def calc_stats(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "Calmar": calmar}


# ── 主流程 ─────────────────────────────────────────────────

print("加载价格数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = MOMENTUM_WINDOW + 20
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

# 加载辅助数据
print("\n加载辅助数据...")
pe_series = load_pe_data()
nb_series = load_northbound_data()

# 预计算
print("\n预计算信号与区制...")
scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

# 方向C：市场状态区制
print("  计算市场状态区制（PE分位+波动率）...")
regime = build_regime(close, pe_series)
bull_days  = (regime == 3).sum()
range_days = (regime == 2).sum()
bear_days  = (regime == 1).sum()
total = len(regime)
print(f"  区制分布：牛市={bull_days/total:.1%}，震荡={range_days/total:.1%}，熊市={bear_days/total:.1%}")

# 方向D：北向资金过滤
print("  计算北向资金过滤信号...")
nb_filter = build_northbound_filter(close, nb_series, NORTHBOUND_MA)
if not nb_series.empty:
    allow_pct = nb_filter.mean()
    print(f"  允许买入交易日占比：{allow_pct:.1%}（禁买期 = {1-allow_pct:.1%}）")

# IS/OOS 分割
n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
print(f"\nIS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")

# 基准
bench = close[BENCHMARK].dropna()
bench_nav = bench / bench.iloc[0] * INIT_CASH

print("\n开始回测（全样本）...")

# ── 配置定义 ──────────────────────────────────────────────
# (label, use_lw, top_n_regime, allow_buy_filter)
CONFIGS = [
    ("基线（风险调整，已上线）",             False, None,   None     ),
    ("方向A: Ledoit-Wolf最小方差权重",       True,  None,   None     ),
    ("方向C: 市场状态区制（PE+波动率）",     False, regime, None     ),
    ("方向D: 北向资金禁买过滤",              False, None,   nb_filter),
    ("方向A+C: LW+区制",                    True,  regime, None     ),
    ("方向A+D: LW+北向",                    True,  None,   nb_filter),
    ("方向C+D: 区制+北向",                  False, regime, nb_filter),
    ("方向A+C+D: 全组合",                   True,  regime, nb_filter),
]

rows = []
navs = {}
for label, use_lw, top_n_ov, nb_flt in CONFIGS:
    nav = run_backtest(
        close, scores, rebal_dates,
        use_ledoit_wolf=use_lw,
        top_n_regime=top_n_ov,
        allow_buy_filter=nb_flt,
    )
    s = calc_stats(nav)
    rows.append({
        "配置":     label,
        "年化收益": f"{s['CAGR']*100:.1f}%",
        "夏普":     f"{s['Sharpe']:.2f}",
        "最大回撤": f"{s['MaxDD']*100:.1f}%",
        "Calmar":   f"{s['Calmar']:.2f}",
    })
    navs[label] = nav
    print(f"  {label:<32}  夏普={s['Sharpe']:.2f}  年化={s['CAGR']*100:.1f}%  回撤={s['MaxDD']*100:.1f}%")

result_df = pd.DataFrame(rows).set_index("配置")
print("\n" + "=" * 75)
print("全样本对比（2016-2026）")
print("=" * 75)
print(result_df.to_string())

# ── IS/OOS 验证（所有配置）──────────────────────────────────

close_is  = close[close.index <  split_date]
close_oos = close[close.index >= split_date]
rebal_is  = [d for d in rebal_dates if d <  split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
sc_is     = scores[scores.index <  split_date]
sc_oos    = scores[scores.index >= split_date]

print(f"\n\nIS/OOS 验证（IS: 2016~{split_date.date()}，OOS: {split_date.date()}~2026）")
print("=" * 75)

is_oos_rows = []
for label, use_lw, top_n_ov, nb_flt in CONFIGS:
    ov_is  = top_n_ov[top_n_ov.index <  split_date] if top_n_ov is not None else None
    ov_oos = top_n_ov[top_n_ov.index >= split_date] if top_n_ov is not None else None
    nf_is  = nb_flt[nb_flt.index <  split_date] if nb_flt is not None else None
    nf_oos = nb_flt[nb_flt.index >= split_date] if nb_flt is not None else None

    n_is  = run_backtest(close_is,  sc_is,  rebal_is,  use_ledoit_wolf=use_lw,
                         top_n_regime=ov_is,  allow_buy_filter=nf_is)
    n_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_ledoit_wolf=use_lw,
                         top_n_regime=ov_oos, allow_buy_filter=nf_oos)

    si = calc_stats(n_is)
    so = calc_stats(n_oos)
    decay = so["Sharpe"] / si["Sharpe"] if si["Sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告:过拟合"

    is_oos_rows.append({
        "配置": label,
        "IS夏普":  f"{si['Sharpe']:.2f}",
        "IS年化":  f"{si['CAGR']*100:.1f}%",
        "IS回撤":  f"{si['MaxDD']*100:.1f}%",
        "OOS夏普": f"{so['Sharpe']:.2f}",
        "OOS年化": f"{so['CAGR']*100:.1f}%",
        "OOS回撤": f"{so['MaxDD']*100:.1f}%",
        "OOS/IS":  f"{decay:.2f}",
        "状态": status,
    })
    print(f"\n{label}")
    print(f"  IS ：夏普={si['Sharpe']:.2f}  年化={si['CAGR']*100:.1f}%  回撤={si['MaxDD']*100:.1f}%")
    print(f"  OOS：夏普={so['Sharpe']:.2f}  年化={so['CAGR']*100:.1f}%  回撤={so['MaxDD']*100:.1f}%")
    print(f"  OOS/IS={decay:.2f}  [{status}]")

is_oos_df = pd.DataFrame(is_oos_rows).set_index("配置")
print("\n\n汇总表：")
print(is_oos_df.to_string())

# ── 净值曲线图 ────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]})
ax1, ax2 = axes

colors = ["#9E9E9E", "#2196F3", "#E53935", "#43A047",
          "#7B1FA2", "#00838F", "#F57F17", "#1A237E"]
for (label, *_), color in zip(CONFIGS, colors):
    nav = navs[label]
    lw = 2.2 if "已上线" in label else 1.4
    ax1.plot(nav.index, nav / INIT_CASH, label=label, color=color, linewidth=lw)

ax1.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800",
         linestyle="--", linewidth=1.2, alpha=0.7, label="沪深300买持")
ax1.axvline(split_date, color="red", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_title("ETF轮动 v3 优化方向净值对比（2016-2026）")
ax1.set_ylabel("净值")
ax1.legend(fontsize=7.5, ncol=2, loc="upper left")
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
ax1.text(split_date, ax1.get_ylim()[0], " IS|OOS", color="red", fontsize=8, va="bottom")

# 回撤对比（基线 vs 最优）
base_nav = navs["基线（风险调整，已上线）"]
base_dd = (base_nav - base_nav.cummax()) / base_nav.cummax() * 100
ax2.fill_between(base_dd.index, base_dd, 0, alpha=0.35, color="#9E9E9E", label="基线回撤")

# 找全样本夏普最高的非基线配置
best_label = max(
    [r["配置"] for r in rows if r["配置"] != "基线（风险调整，已上线）"],
    key=lambda l: float(next(r["夏普"] for r in rows if r["配置"] == l)),
)
best_nav = navs[best_label]
best_dd = (best_nav - best_nav.cummax()) / best_nav.cummax() * 100
ax2.fill_between(best_dd.index, best_dd, 0, alpha=0.35, color="#2196F3", label=f"最优({best_label[:8]}...) 回撤")
ax2.set_ylabel("回撤(%)")
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = out_dir / "etf_rotation_v3_compare.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n净值曲线图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
