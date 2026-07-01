"""
ETF 轮动月度信号
每月第一个交易日运行，输出本月持仓建议。
参数：Top3，动量窗口25日，风险调整动量（OLS斜率×R²÷波动率）
拥挤度修正：行业拥挤度分位数 > 0.75 时，动量得分 × 0.2
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, run_update
from etf_universe import ETF_UNIVERSE

MOMENTUM_WINDOW    = 25
RISK_VOL_WINDOW    = 21    # 风险调整动量：除以近N日年化波动率
TOP_N              = 3
CORR_WINDOW        = 60    # 拥挤度相关系数计算窗口
CORR_HIST_WINDOW   = 252   # 拥挤度历史分位数窗口
CROWD_THRESHOLD    = 0.75  # 拥挤度分位数阈值
CROWD_FACTOR       = 0.2   # 超过阈值时动量得分乘以此系数
BENCHMARK          = "510300.SH"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
SIGNAL_LOG = pathlib.Path(__file__).parent / "results" / "signal_log.csv"


# ── 动量评分 ──────────────────────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + _
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def get_scores_today(close: pd.DataFrame, window: int, vol_window: int) -> pd.Series:
    """
    计算最新交易日各标的风险调整动量得分。
    得分 = OLS斜率×R²÷近vol_window日年化波动率
    """
    scores = {}
    for code in close.columns:
        series = close[code].dropna()
        if len(series) < window + vol_window:
            continue
        raw = momentum_score(series.iloc[-window:])
        ret = series.iloc[-vol_window:].pct_change().dropna()
        vol = ret.std() * np.sqrt(252)
        scores[code] = raw / vol if vol > 1e-6 else raw
    return pd.Series(scores).sort_values(ascending=False)


def get_crowding_today(
    close: pd.DataFrame,
    corr_window: int = CORR_WINDOW,
    hist_window: int = CORR_HIST_WINDOW,
) -> pd.Series:
    """
    计算当日各ETF拥挤度历史分位数（0~1）。
    拥挤度 = 该ETF与宇宙内其余ETF的平均两两相关系数（滚动60日）。
    返回：各ETF的拥挤度分位数，index 为代码。
    """
    codes = [c for c in close.columns if c != BENCHMARK]
    needed = corr_window + hist_window
    if len(close) < needed:
        return pd.Series(dtype=float)

    ret = close[codes].pct_change()

    # 计算最近 hist_window + 1 日（足够估算历史分位）的滚动拥挤度
    start_i = len(close) - hist_window - 1
    crowding_raw = {}

    for i in range(start_i, len(close)):
        ret_win = ret.iloc[i - corr_window: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(np.nanmean(corr_arr, axis=1), index=ret_win.columns)
        crowding_raw[close.index[i]] = avg_corr

    if not crowding_raw:
        return pd.Series(dtype=float)

    raw_df = pd.DataFrame(crowding_raw).T  # shape: (hist_window+1, n_codes)

    # 最新一行 vs 历史（去掉最后一行）
    hist = raw_df.iloc[:-1]
    curr = raw_df.iloc[-1]

    pct = {}
    for code in codes:
        h = hist[code].dropna() if code in hist.columns else pd.Series()
        c = curr.get(code, np.nan)
        if pd.isna(c) or len(h) < 20:
            pct[code] = np.nan
        else:
            pct[code] = (h < c).mean()

    return pd.Series(pct)


# ── 读取上月持仓 ──────────────────────────────────────────

def load_last_holdings() -> list[str]:
    if not SIGNAL_LOG.exists():
        return []
    log = pd.read_csv(SIGNAL_LOG)
    if log.empty:
        return []
    last = log.iloc[-1]
    holdings = []
    for col in ["持仓1", "持仓2", "持仓3"]:
        if col in last and pd.notna(last[col]) and last[col] != "现金":
            holdings.append(last[col])
    return holdings


def save_signal(date: str, holdings: list[str], scores: pd.Series) -> None:
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "日期": date,
        "持仓1": holdings[0] if len(holdings) > 0 else "现金",
        "持仓2": holdings[1] if len(holdings) > 1 else "现金",
        "持仓3": holdings[2] if len(holdings) > 2 else "现金",
        "得分1": f"{scores.iloc[0]:.3f}" if len(scores) > 0 else "",
        "得分2": f"{scores.iloc[1]:.3f}" if len(scores) > 1 else "",
        "得分3": f"{scores.iloc[2]:.3f}" if len(scores) > 2 else "",
    }
    if SIGNAL_LOG.exists():
        log = pd.read_csv(SIGNAL_LOG)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(SIGNAL_LOG, index=False)


# ── 主流程 ────────────────────────────────────────────────

def main():
    # 1. 更新数据到最新
    print("更新行情数据...")
    run_update()

    # 2. 加载收盘价
    close = load_close_matrix()
    today = close.index[-1]
    print(f"最新数据日期：{today.date()}\n")

    # 3. 计算动量得分
    scores = get_scores_today(close, MOMENTUM_WINDOW, RISK_VOL_WINDOW)

    # 4. 计算拥挤度分位数并修正得分
    print("计算行业拥挤度...")
    crowding = get_crowding_today(close)
    adj_scores = scores.copy()
    crowded_codes = []
    for code in adj_scores.index:
        pct = crowding.get(code, np.nan)
        if not pd.isna(pct) and pct > CROWD_THRESHOLD:
            adj_scores[code] *= CROWD_FACTOR
            crowded_codes.append((code, pct))

    pos_scores = adj_scores[adj_scores > 0]

    # 5. 生成信号
    target = list(pos_scores.nlargest(TOP_N).index)
    last_holdings = load_last_holdings()

    # 6. 输出结果
    print("=" * 60)
    print(f"ETF 轮动月度信号  {today.date()}")
    print(f"（拥挤度修正：threshold={CROWD_THRESHOLD}, factor={CROWD_FACTOR}）")
    print("=" * 60)

    if not target:
        print("本月信号：全部空仓（无正动量标的）")
        print("操作：卖出所有持仓，持现金")
    else:
        print(f"本月目标持仓（等权，各约{100//len(target)}%）：")
        for i, code in enumerate(target, 1):
            name = ETF_UNIVERSE.get(code, code)
            score_raw = scores.get(code, 0)
            score_adj = adj_scores[code]
            pct = crowding.get(code, np.nan)
            crowd_str = f"  拥挤度分位={pct:.2f}" if not pd.isna(pct) else ""
            marker = " ← 新增" if code not in last_holdings else ""
            print(f"  {i}. {code}  {name:<16}  原始={score_raw:.3f} 调整后={score_adj:.3f}{crowd_str}{marker}")

    # 7. 调仓操作
    to_sell = [c for c in last_holdings if c not in target]
    to_buy  = [c for c in target if c not in last_holdings]

    print()
    if not last_holdings:
        print("上月持仓：无记录（首次运行）")
    else:
        print(f"上月持仓：{', '.join(last_holdings)}")

    print()
    if to_sell:
        print("需要卖出：")
        for code in to_sell:
            print(f"  - {code}  {ETF_UNIVERSE.get(code, code)}")
    if to_buy:
        print("需要买入：")
        for code in to_buy:
            print(f"  + {code}  {ETF_UNIVERSE.get(code, code)}")
    if not to_sell and not to_buy and target:
        print("无需调仓（持仓未变化）")

    # 8. 拥挤度预警（超过阈值的标的）
    if crowded_codes:
        print()
        print(f"── 拥挤度预警（分位数>{CROWD_THRESHOLD}，得分已打折）" + "─" * 20)
        for code, pct in sorted(crowded_codes, key=lambda x: -x[1]):
            name = ETF_UNIVERSE.get(code, code)
            print(f"  {code}  {name:<16}  拥挤度分位={pct:.2f}")

    # 9. 完整得分排行（前10，显示调整后得分）
    print()
    print("── 动量得分排行（前10，已含拥挤度修正）" + "─" * 20)
    top10 = adj_scores.sort_values(ascending=False).head(10)
    for code, score in top10.items():
        name = ETF_UNIVERSE.get(code, code)
        flag = "✓" if code in target else " "
        trend = "↑" if score > 0 else "↓"
        pct = crowding.get(code, np.nan)
        crowd_str = f"  [{pct:.2f}]" if not pd.isna(pct) else ""
        print(f"  {flag} {code}  {name:<16}  {trend} {score:+.3f}{crowd_str}")

    # 10. 保存信号记录
    target_scores = pos_scores.nlargest(TOP_N) if target else pd.Series(dtype=float)
    save_signal(str(today.date()), target, target_scores)
    print(f"\n信号已记录：{SIGNAL_LOG}")


if __name__ == "__main__":
    main()
