"""
ETF 轮动新优化方向测试（2026-07）
基于调研结论，测试以下 P1/P2 方向：
  P1-a: skip-month（窗口25日但跳过近21日，规避A股短期反转）
  P1-b: 绝对动量过滤（6个月 Dual Momentum）
  P1-c: 空仓切国债ETF（替代货币ETF，熊市有超额收益）
  P2-a: 动态持仓数（20日波动率区制：低→Top3，中→Top2，高→Top1）
  P2-b: 广度连续仓位（正动量ETF占比线性决定总仓位0~100%）
  P2-c: 多周期复合信号（25+63+126日等权合成）
  组合: 绝对动量 + 国债ETF
  组合: 绝对动量 + 动态持仓数
输出：全样本对比表 + 夏普Top3的IS/OOS验证
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
from etf_universe import ETF_UNIVERSE

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH       = 1_000_000
COMMISSION      = 0.0001
SLIPPAGE        = 0.0002
BENCHMARK       = "510300.SH"
START_DATE      = "2016-01-01"
IS_RATIO        = 0.8
MOMENTUM_WINDOW = 25
TOP_N           = 3
RISK_VOL_WINDOW = 21
ABS_MOM_WINDOW  = 126    # 6个月 ≈ 126交易日
CASH_ETF_MONEY  = "511880.SH"   # 货币ETF（空仓停泊，已有数据）
CASH_ETF_BOND   = "511010.SH"   # 国债ETF（熊市对冲）

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 信号计算 ───────────────────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    """OLS斜率×R² 年化动量得分"""
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
    window: int,
    risk_adj: bool = True,
    risk_vol_window: int = RISK_VOL_WINDOW,
    skip_days: int = 0,
) -> pd.DataFrame:
    """
    计算动量得分矩阵。
    skip_days > 0：跳过最近 skip_days 日，规避短期反转。
    risk_adj=True：得分 ÷ 近期年化波动率（风险调整动量）。
    """
    scores = {}
    lookback = window + skip_days
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(lookback, len(series)):
            pw = series.iloc[i - lookback: i - skip_days] if skip_days > 0 else series.iloc[i - window: i]
            raw = momentum_score(pw)
            if risk_adj and i >= risk_vol_window:
                rets = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            score_series.iloc[i] = raw
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


def calc_composite_scores(
    close_matrix: pd.DataFrame,
    windows: list = None,
    risk_adj: bool = True,
) -> pd.DataFrame:
    """
    多周期等权合成动量信号。
    对各窗口得分等权平均（NaN不参与平均），保留绝对值（用于正负判断）。
    """
    if windows is None:
        windows = [25, 63, 126]
    frames = []
    for w in windows:
        sc = calc_all_scores(close_matrix, w, risk_adj=risk_adj)
        frames.append(sc)
    # 逐元素平均（忽略NaN）
    stacked = pd.concat(frames, axis=0)
    grouped = stacked.groupby(level=0).mean()
    return grouped.reindex(close_matrix.index)


def precompute_vol_regime(close: pd.DataFrame, vol_window: int = 20, hist_lookback: int = 252) -> pd.Series:
    """
    预计算每个交易日沪深300的波动率区制。
    返回 pd.Series，values: top_n_override (1/2/3)，index=date
    """
    rets = close[BENCHMARK].pct_change()
    vol_20 = rets.rolling(vol_window).std() * np.sqrt(252)
    regime = pd.Series(TOP_N, index=close.index, dtype=int)
    for i in range(hist_lookback + vol_window, len(close.index)):
        date = close.index[i]
        curr_vol = vol_20.iloc[i]
        if pd.isna(curr_vol):
            continue
        hist = vol_20.iloc[i - hist_lookback: i].dropna()
        if len(hist) < 30:
            continue
        pct33 = hist.quantile(0.33)
        pct67 = hist.quantile(0.67)
        if curr_vol <= pct33:
            regime[date] = 3   # 低波动：Top3
        elif curr_vol <= pct67:
            regime[date] = 2   # 中波动：Top2
        else:
            regime[date] = 1   # 高波动：Top1
    return regime


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 回测主逻辑（扩展版）────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    cash_etf: str = None,
    # 新增参数
    use_abs_momentum: bool = False,
    abs_mom_window: int = ABS_MOM_WINDOW,
    top_n_override: pd.Series = None,   # 动态持仓数（预计算的区制序列）
    breadth_continuous: bool = False,    # 广度连续仓位
) -> pd.Series:
    """
    月度轮换回测，支持以下扩展：
    - use_abs_momentum: 绝对动量过滤（6M return > 0 才入选）
    - top_n_override: 动态持仓数（按波动率区制预计算）
    - breadth_continuous: 广度连续仓位（正动量占比→总仓位0~100%）
    """
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

        # 当日有效得分（排除空仓停泊ETF）
        day_scores = scores.loc[date].dropna()
        if cash_etf:
            day_scores = day_scores.drop(labels=[cash_etf], errors="ignore")

        # 动态持仓数
        effective_top_n = int(top_n_override[date]) if (top_n_override is not None and date in top_n_override.index) else top_n

        # 初筛：正动量候选
        pos_scores = day_scores[day_scores > 0].nlargest(effective_top_n * 3)
        candidates = list(pos_scores.index)

        # 绝对动量过滤（6M return > 0）
        if use_abs_momentum and candidates:
            valid = []
            for code in candidates:
                if code not in close.columns:
                    continue
                series = close[code].dropna()
                try:
                    loc = series.index.get_loc(date)
                except KeyError:
                    continue
                if loc >= abs_mom_window:
                    abs_ret = series.iloc[loc] / series.iloc[loc - abs_mom_window] - 1
                    if abs_ret > 0:
                        valid.append(code)
            candidates = valid

        target_codes = candidates[:effective_top_n]

        # 广度连续仓位：正动量ETF占比→总仓位比例
        if breadth_continuous:
            n_pos = (day_scores > 0).sum()
            breadth = n_pos / max(len(day_scores), 1)
            # breadth < 0.25 → 0%, breadth > 0.55 → 100%，线性插值
            total_alloc = min(1.0, max(0.0, (breadth - 0.25) / 0.30))
        else:
            total_alloc = 1.0

        # 卖出不在目标中的持仓
        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            # 无正动量：停泊到防御性资产
            if cash_etf and cash_etf in close.columns:
                price = close.loc[date, cash_etf]
                if pd.notna(price) and cash > price:
                    buy_price = price * (1 + SLIPPAGE / 2)
                    buy_shares = int(cash / buy_price / 100) * 100
                    if buy_shares > 0:
                        cost = buy_shares * buy_price * (1 + COMMISSION)
                        if cash >= cost:
                            cash -= cost
                            holdings[cash_etf] = holdings.get(cash_etf, 0) + buy_shares
            continue

        # 等权分配，按 total_alloc 缩放目标仓位
        n = len(target_codes)
        weights = {c: 1.0 / n for c in target_codes}

        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            buy_price = price * (1 + SLIPPAGE / 2)
            target_value = port_value * weights[code] * total_alloc
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
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd,
            "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0}


# ── 主流程 ────────────────────────────────────────────────

def load_defensive_etf(code: str, close: pd.DataFrame) -> pd.DataFrame:
    """加载防御性资产ETF（国债/货币），附加到 close 矩阵"""
    path = pathlib.Path(__file__).parent.parent.parent / "data" / "daily" / f"{code}.parquet"
    if not path.exists():
        print(f"  [跳过] {code} 数据文件不存在，相关配置将回退到货币ETF")
        return close
    df = pd.read_parquet(path, columns=["trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    series = df.set_index("trade_date")["close"].reindex(close.index)
    close = close.copy()
    close[code] = series
    print(f"  [加载] {code} 防御性资产已附加")
    return close


print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
min_records = 126 + 30   # 最长窗口（多周期复合126日）
valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
close = close[valid_codes]

# 加载防御性资产（不在 ETF_UNIVERSE 轮动池中）
close = load_defensive_etf(CASH_ETF_MONEY, close)
close = load_defensive_etf(CASH_ETF_BOND, close)

# 国债ETF不可用时回退到货币ETF
BOND_ETF_AVAILABLE = CASH_ETF_BOND in close.columns
if not BOND_ETF_AVAILABLE:
    print(f"  [提示] 国债ETF {CASH_ETF_BOND} 不可用，相关配置使用货币ETF替代")
    CASH_ETF_BOND_ACTUAL = CASH_ETF_MONEY
else:
    CASH_ETF_BOND_ACTUAL = CASH_ETF_BOND

print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} → {close.index[-1].date()}")

# IS/OOS 分割
n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
print(f"IS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} / {split_date.date()} ~ {close.index[-1].date()}")

# 基准净值
bench = close[BENCHMARK].dropna()
bench_nav = bench / bench.iloc[0] * INIT_CASH

# 调仓日
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

print("\n预计算各信号得分（需要几分钟）...")

# 各版本得分（全量）
sc_base    = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)           # 基线（已上线）
sc_skip21  = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True, skip_days=21)  # skip-month
sc_comp    = calc_composite_scores(close, windows=[25, 63, 126], risk_adj=True)    # 多周期复合

# 动态持仓数：预计算波动率区制
print("预计算波动率区制...")
vol_regime = precompute_vol_regime(close)

print("信号预计算完成，开始回测...\n")

# ── 配置定义 ──────────────────────────────────────────────
# 每条配置：(label, scores, cash_etf, use_abs_mom, top_n_override, breadth_continuous)
CONFIGS = [
    ("基线（风险调整，已上线）",      sc_base,   CASH_ETF_MONEY,       False, None,       False),
    ("+skip21日（规避短期反转）",     sc_skip21, CASH_ETF_MONEY,       False, None,       False),
    ("+绝对动量6M过滤",              sc_base,   CASH_ETF_MONEY,       True,  None,       False),
    ("+空仓切国债ETF",               sc_base,   CASH_ETF_BOND_ACTUAL, False, None,       False),
    ("+动态持仓数（区制Top3/2/1）",  sc_base,   CASH_ETF_MONEY,       False, vol_regime, False),
    ("+广度连续仓位",                sc_base,   CASH_ETF_MONEY,       False, None,       True ),
    ("+多周期复合信号(25+63+126)",   sc_comp,   CASH_ETF_MONEY,       False, None,       False),
    ("+绝对动量+国债ETF",            sc_base,   CASH_ETF_BOND_ACTUAL, True,  None,       False),
    ("+绝对动量+动态持仓数",         sc_base,   CASH_ETF_MONEY,       True,  vol_regime, False),
    ("+绝对动量+多周期+国债",        sc_comp,   CASH_ETF_BOND_ACTUAL, True,  None,       False),
]

# ── 全样本对比 ────────────────────────────────────────────

rows = []
navs = {}
for label, sc, cef, use_abs, top_n_ov, breadth_c in CONFIGS:
    nav = run_backtest(close, sc, rebal_dates,
                       cash_etf=cef,
                       use_abs_momentum=use_abs,
                       top_n_override=top_n_ov,
                       breadth_continuous=breadth_c)
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

# ── IS/OOS 验证（夏普Top3配置）────────────────────────────

top3_labels = sorted(navs, key=lambda l: float(rows[[r["配置"] for r in rows].index(l)]["夏普"]), reverse=True)[:3]

close_is  = close[close.index < split_date]
close_oos = close[close.index >= split_date]
rebal_is  = [d for d in rebal_dates if d < split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]

print(f"\n\n夏普Top3配置 IS/OOS 验证")
print("=" * 75)

for label in top3_labels:
    cfg = next(c for c in CONFIGS if c[0] == label)
    _, sc_full, cef, use_abs, top_n_ov, breadth_c = cfg

    # IS/OOS 得分截取
    sc_is  = sc_full[sc_full.index <  split_date] if sc_full is not None else None
    sc_oos = sc_full[sc_full.index >= split_date] if sc_full is not None else None

    # top_n_override 截取
    ov_is  = top_n_ov[top_n_ov.index <  split_date] if top_n_ov is not None else None
    ov_oos = top_n_ov[top_n_ov.index >= split_date] if top_n_ov is not None else None

    n_is  = run_backtest(close_is,  sc_is,  rebal_is,  cash_etf=cef, use_abs_momentum=use_abs,
                         top_n_override=ov_is,  breadth_continuous=breadth_c)
    n_oos = run_backtest(close_oos, sc_oos, rebal_oos, cash_etf=cef, use_abs_momentum=use_abs,
                         top_n_override=ov_oos, breadth_continuous=breadth_c)

    si  = calc_stats(n_is)
    so  = calc_stats(n_oos)
    decay = so["Sharpe"] / si["Sharpe"] if si["Sharpe"] > 0 else 0
    status = "通过" if decay >= 0.5 else "警告：可能过拟合"

    print(f"\n{label}")
    print(f"  IS  夏普={si['Sharpe']:.2f}  年化={si['CAGR']*100:.1f}%  回撤={si['MaxDD']*100:.1f}%")
    print(f"  OOS 夏普={so['Sharpe']:.2f}  年化={so['CAGR']*100:.1f}%  回撤={so['MaxDD']*100:.1f}%")
    print(f"  OOS/IS 夏普比={decay:.2f}  [{status}]")

# ── 净值曲线图 ────────────────────────────────────────────

out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8))
colors = ["#9E9E9E", "#2196F3", "#E53935", "#43A047", "#FB8C00",
          "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41", "#1E88E5"]
for (label, *_), color in zip(CONFIGS, colors):
    nav = navs[label]
    lw = 2.0 if label == "基线（风险调整，已上线）" else 1.3
    ax.plot(nav.index, nav / INIT_CASH, label=label, color=color, linewidth=lw)
ax.plot(bench_nav.index, bench_nav / INIT_CASH, color="#FF9800",
        linestyle="--", linewidth=1.2, alpha=0.7, label="沪深300买持")
ax.axvline(split_date, color="red", linestyle="--", alpha=0.6, linewidth=1)
ax.set_title("ETF轮动新优化方向净值对比（2016-2026）")
ax.set_ylabel("净值")
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.grid(alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
plt.tight_layout()
fig_path = out_dir / "etf_rotation_v2_compare.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n净值曲线图已保存：{fig_path}")
plt.close("all")

print("\n完成。")
