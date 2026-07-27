"""
第十轮：组合波动率目标控制方向 convergence 复核（2026-07）

背景：v9_voltarget.py 用人工标准判定"不通过稳健性验证"（网格仅25%超基线，
滚动3年43.8%劣于基线）。这里给网格中每组参数都补算IS/OOS夏普，导出CSV
交给 convergence skill 用数值化标准交叉复核。
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix

INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21

TARGET_VOL_GRID  = [0.12, 0.15, 0.18, 0.22]
VOL_LOOKBACK_GRID = [21, 42, 63]
MIN_EXPOSURE     = 0.3
MAX_EXPOSURE     = 1.0


def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
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
                  use_vol_target=False, target_vol=0.15, vol_lookback=21) -> pd.Series:
    cash = init_cash
    holdings: dict = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)
    nav_hist: list = []

    for date in close.index:
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value
        nav_hist.append(port_value)

        if date not in rebal_set:
            continue

        day_scores = scores.loc[date].dropna()
        pos_scores = day_scores[day_scores > 0].nlargest(top_n * 3)
        candidates = list(pos_scores.index)
        target_codes = candidates[:top_n]

        exposure = 1.0
        if use_vol_target and len(nav_hist) >= vol_lookback + 1:
            recent_nav = pd.Series(nav_hist[-(vol_lookback + 1):])
            recent_rets = recent_nav.pct_change().dropna()
            realized_vol = recent_rets.std() * np.sqrt(252)
            if realized_vol > 1e-6:
                exposure = target_vol / realized_vol
                exposure = float(np.clip(exposure, MIN_EXPOSURE, MAX_EXPOSURE))

        for code in list(holdings.keys()):
            if code not in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not target_codes:
            continue

        n = len(target_codes)
        weights = {c: exposure / n for c in target_codes}

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

nav_base_full = run_backtest(close, scores, rebal_dates, use_vol_target=False)
nav_base_is = run_backtest(close_is, sc_is, rebal_is, use_vol_target=False)
nav_base_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_vol_target=False)
base_sharpe_full = sharpe_of(nav_base_full)
base_sharpe_is = sharpe_of(nav_base_is)
base_sharpe_oos = sharpe_of(nav_base_oos)
print(f"基线：全样本={base_sharpe_full:.3f}  IS={base_sharpe_is:.3f}  OOS={base_sharpe_oos:.3f}")

rows = [{
    "variant_id": "baseline",
    "target_vol": 0,
    "vol_lookback": 0,
    "sharpe_full": base_sharpe_full,
    "sharpe_is": base_sharpe_is,
    "sharpe_oos": base_sharpe_oos,
    "n_trades": len(rebal_dates),
}]

print("\n网格扫描 × IS/OOS（每组参数单独跑IS和OOS）...")
for tv in TARGET_VOL_GRID:
    for lb in VOL_LOOKBACK_GRID:
        nav_full = run_backtest(close, scores, rebal_dates, use_vol_target=True,
                                 target_vol=tv, vol_lookback=lb)
        nav_is = run_backtest(close_is, sc_is, rebal_is, use_vol_target=True,
                               target_vol=tv, vol_lookback=lb)
        nav_oos = run_backtest(close_oos, sc_oos, rebal_oos, use_vol_target=True,
                                target_vol=tv, vol_lookback=lb)
        s_full, s_is, s_oos = sharpe_of(nav_full), sharpe_of(nav_is), sharpe_of(nav_oos)
        rows.append({
            "variant_id": f"tv{int(tv*100)}_lb{lb}",
            "target_vol": tv,
            "vol_lookback": lb,
            "sharpe_full": s_full,
            "sharpe_is": s_is,
            "sharpe_oos": s_oos,
            "n_trades": len(rebal_dates),
        })
        print(f"  tv={tv:.0%}  lb={lb:<3} 全样本={s_full:.3f}  IS={s_is:.3f}  OOS={s_oos:.3f}")

df = pd.DataFrame(rows)
out_dir = pathlib.Path(__file__).parent.parent / "results"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "v10_voltarget_convergence.csv"
df.to_csv(out_path, index=False)
print(f"\nCSV已保存：{out_path}")
