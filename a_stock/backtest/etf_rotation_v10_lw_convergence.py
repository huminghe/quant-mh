"""
第十轮：LW最小方差方向 convergence 复核（2026-07）

背景：v3c_lw_fullval.py 用人工标准（网格超基线比例/逐年拆解/滚动3年/IS-OOS单点）
判定LW方向"不通过稳健性验证"。这里给网格中每组参数都补算IS/OOS夏普，
导出CSV交给 convergence skill（martianmobile/strategy-evaluation）用数值化标准
（Top-K离散度 + IS/OOS Spearman + 参数平台检测）交叉复核，而非重新假设结论。
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix

INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21

MIN_HISTORY_GRID = [30, 60, 90, 120]
HIST_WINDOW_GRID = [126, 189, 252]


def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix: pd.DataFrame) -> pd.DataFrame:
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


def run_backtest(close, scores, rebal_dates, top_n=TOP_N, init_cash=INIT_CASH,
                  use_ledoit_wolf=False, lw_min_history=60, hist_window=252) -> pd.Series:
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
        pos_scores = day_scores[day_scores > 0].nlargest(top_n * 3)
        candidates = list(pos_scores.index)
        target_codes = candidates[:top_n]

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        n = len(target_codes)
        if use_ledoit_wolf and n >= 2:
            date_loc = close.index.get_loc(date)
            hist_start = max(0, date_loc - hist_window)
            ret_hist = close[target_codes].iloc[hist_start:date_loc].pct_change().dropna()
            if len(ret_hist) >= lw_min_history:
                try:
                    lw = LedoitWolf().fit(ret_hist.values)
                    cov = lw.covariance_
                    ones = np.ones(n)
                    inv_cov = np.linalg.pinv(cov)
                    raw_w = inv_cov @ ones
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

        for code in target_codes:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
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


def sharpe_of(nav: pd.Series) -> float:
    r = nav.pct_change().dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


print("加载数据...")
close_full = load_close_matrix()
close = close_full[close_full.index >= START_DATE]
valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
close = close[valid_codes]
print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

print("计算动量得分...")
scores = calc_all_scores(close)
rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

n_days = len(close)
split_idx = int(n_days * IS_RATIO)
split_date = close.index[split_idx]
close_is, close_oos = close[close.index < split_date], close[close.index >= split_date]
rebal_is = [d for d in rebal_dates if d < split_date]
rebal_oos = [d for d in rebal_dates if d >= split_date]
sc_is, sc_oos = scores[scores.index < split_date], scores[scores.index >= split_date]

print(f"IS: {close_is.index[0].date()}~{close_is.index[-1].date()}  "
      f"OOS: {close_oos.index[0].date()}~{close_oos.index[-1].date()}")

nav_base_full = run_backtest(close, scores, rebal_dates, use_ledoit_wolf=False)
nav_base_is = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=False)
nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_ledoit_wolf=False)
base_sharpe_full = sharpe_of(nav_base_full)
base_sharpe_is = sharpe_of(nav_base_is)
base_sharpe_oos = sharpe_of(nav_base_oos)
print(f"基线：全样本={base_sharpe_full:.3f}  IS={base_sharpe_is:.3f}  OOS={base_sharpe_oos:.3f}")

rows = [{
    "variant_id": "baseline",
    "lw_min_history": 0,
    "hist_window": 0,
    "sharpe_full": base_sharpe_full,
    "sharpe_is": base_sharpe_is,
    "sharpe_oos": base_sharpe_oos,
    "n_trades": len(rebal_dates),
}]

print("\n网格扫描 × IS/OOS（每组参数单独跑IS和OOS）...")
for mh in MIN_HISTORY_GRID:
    for hw in HIST_WINDOW_GRID:
        if mh > hw:
            continue
        nav_full = run_backtest(close, scores, rebal_dates, use_ledoit_wolf=True,
                                 lw_min_history=mh, hist_window=hw)
        nav_is = run_backtest(close_is, sc_is, rebal_is, use_ledoit_wolf=True,
                               lw_min_history=mh, hist_window=hw)
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_ledoit_wolf=True,
                                lw_min_history=mh, hist_window=hw)
        s_full, s_is, s_oos = sharpe_of(nav_full), sharpe_of(nav_is), sharpe_of(nav_oos)
        rows.append({
            "variant_id": f"mh{mh}_hw{hw}",
            "lw_min_history": mh,
            "hist_window": hw,
            "sharpe_full": s_full,
            "sharpe_is": s_is,
            "sharpe_oos": s_oos,
            "n_trades": len(rebal_dates),
        })
        print(f"  mh={mh:<4} hw={hw:<4} 全样本={s_full:.3f}  IS={s_is:.3f}  OOS={s_oos:.3f}")

df = pd.DataFrame(rows)
out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "v10_lw_convergence.csv"
df.to_csv(out_path, index=False)
print(f"\nCSV已保存：{out_path}")
