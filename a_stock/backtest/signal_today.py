"""
ETF 轮动月度信号
每月第一个交易日运行，输出本月持仓建议。
参数：Top3，动量窗口25日，风险调整动量（OLS斜率×R²÷波动率）

注：2026-07-08修复历史价格复权断层bug后重新验证，行业拥挤度软过滤在干净数据上
边际收益基本消失（+0.007夏普，统计不显著），已从信号中移除，简化为纯风险调整动量。
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
    pos_scores = scores[scores > 0]

    # 4. 生成信号
    target = list(pos_scores.nlargest(TOP_N).index)
    last_holdings = load_last_holdings()

    # 5. 输出结果
    print("=" * 60)
    print(f"ETF 轮动月度信号  {today.date()}")
    print("=" * 60)

    if not target:
        print("本月信号：全部空仓（无正动量标的）")
        print("操作：卖出所有持仓，持现金")
    else:
        print(f"本月目标持仓（等权，各约{100//len(target)}%）：")
        for i, code in enumerate(target, 1):
            name = ETF_UNIVERSE.get(code, code)
            score = scores.get(code, 0)
            marker = " ← 新增" if code not in last_holdings else ""
            print(f"  {i}. {code}  {name:<16}  得分={score:.3f}{marker}")

    # 6. 调仓操作
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

    # 7. 完整得分排行（前10）
    print()
    print("── 动量得分排行（前10）" + "─" * 20)
    top10 = scores.sort_values(ascending=False).head(10)
    for code, score in top10.items():
        name = ETF_UNIVERSE.get(code, code)
        flag = "✓" if code in target else " "
        trend = "↑" if score > 0 else "↓"
        print(f"  {flag} {code}  {name:<16}  {trend} {score:+.3f}")

    # 8. 保存信号记录
    target_scores = pos_scores.nlargest(TOP_N) if target else pd.Series(dtype=float)
    save_signal(str(today.date()), target, target_scores)
    print(f"\n信号已记录：{SIGNAL_LOG}")


if __name__ == "__main__":
    main()
