"""
ETF 轮动月度信号
每月第一个交易日运行，输出本月持仓建议。
参数：Top3，动量窗口25日，风险调整动量（OLS斜率×R²÷波动率）+ flow信号连续打折

注：2026-07-08修复历史价格复权断层bug后重新验证，行业拥挤度软过滤在干净数据上
边际收益基本消失（+0.007夏普，统计不显著），已从信号中移除，简化为纯风险调整动量。

2026-07-27起：标的池从45只手工池（etf_universe.ETF_UNIVERSE，已因集中度风险在
模拟盘暴露问题而放弃）切换为431只机械化候选池（`etf_all_candidates.parquet`，
纯滚动126日成交额规则，point-in-time构建），并叠加flow信号（ETF份额月度变化率，
份额净流出→次月表现更好的反转逻辑）。叠加方式为"连续打折"：
day_scores *= (0.5 + boost)，boost为flow的1-rank(pct=True)，与etf_rotation_v38_
fundamental_signal_ablation.py验证的公式完全一致。历史背书：431池2016-2026
全样本回测夏普0.590 vs 纯动量基线0.526，已通过滚动2年窗口稳健性检验
（均值Δ+0.161，劣于基线占比20.7%）。上线后不再重启模拟盘观察，直接用小仓位
实盘验证，因回测背书已覆盖全样本+滚动窗口，重复模拟盘验证边际价值低。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, run_update, init_pro

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all

MOMENTUM_WINDOW    = 25
RISK_VOL_WINDOW    = 21    # 风险调整动量：除以近N日年化波动率
TOP_N              = 3
FLOW_LOOKBACK_MONTHS = 14  # flow信号回看窗口，够算月度变化率即可，不用拉全历史
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
CANDIDATES_FILE = DATA_DIR / "etf_all_candidates.parquet"
BENCHMARK_FILE = DATA_DIR / "etf_benchmark.parquet"
SIGNAL_LOG = pathlib.Path(__file__).parent / "results" / "signal_log.csv"


def load_candidate_codes() -> list:
    """431只机械化候选池标的清单"""
    return pd.read_parquet(CANDIDATES_FILE)["ts_code"].tolist()


def load_etf_names() -> dict:
    """431池ETF中文名称（复用fetch_etf_benchmark.py已拉取的全市场ETF基础信息）"""
    bench = pd.read_parquet(BENCHMARK_FILE)
    return dict(zip(bench["ts_code"], bench["name"]))


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


# ── flow信号（ETF份额月度变化率）──────────────────────────

def get_flow_boost_today(codes: list) -> pd.Series:
    """
    拉取最新ETF份额数据，计算最新一期flow信号的连续打折boost。
    boost = 1 - rank(pct=True)（净流出即份额环比下降→boost更高，与v38验证方向一致）。
    只拉近FLOW_LOOKBACK_MONTHS个月，够算月度变化率，不需要全历史。
    """
    pro = init_pro()
    start = (pd.Timestamp.today() - pd.DateOffset(months=FLOW_LOOKBACK_MONTHS)).strftime("%Y%m%d")
    share_matrix = fetch_fund_share_all(pro, codes, start_date=start)
    if share_matrix.empty:
        return pd.Series(dtype=float)
    monthly_share = share_matrix.resample("ME").last()
    flow_1m = monthly_share.pct_change()
    if flow_1m.empty:
        return pd.Series(dtype=float)
    latest = flow_1m.iloc[-1].dropna()
    if len(latest) < 5:
        return pd.Series(dtype=float)
    r = latest.rank(pct=True)
    return 1 - r  # invert=True：份额净流出排名靠前


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
    candidates = load_candidate_codes()
    names = load_etf_names()

    # 1. 更新数据到最新（431只机械化候选池）
    print(f"更新行情数据（候选池 {len(candidates)} 只）...")
    run_update(codes=candidates)

    # 2. 加载收盘价
    close = load_close_matrix(codes=candidates)
    today = close.index[-1]
    print(f"最新数据日期：{today.date()}\n")

    # 3. 计算动量得分
    scores = get_scores_today(close, MOMENTUM_WINDOW, RISK_VOL_WINDOW)

    # 3.5 叠加flow信号（连续打折，与v38验证公式一致）
    print("拉取flow信号（ETF份额月度变化率）...")
    boost = get_flow_boost_today(list(scores.index))
    if not boost.empty:
        adj_scores = scores.copy()
        for code in adj_scores.index:
            b = boost.get(code)
            if pd.notna(b):
                adj_scores[code] *= (0.5 + b)
        scores = adj_scores.sort_values(ascending=False)
    else:
        print("flow信号拉取为空，本次仅用纯动量（不叠加boost）")
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
            name = names.get(code, code)
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
            print(f"  - {code}  {names.get(code, code)}")
    if to_buy:
        print("需要买入：")
        for code in to_buy:
            print(f"  + {code}  {names.get(code, code)}")
    if not to_sell and not to_buy and target:
        print("无需调仓（持仓未变化）")

    # 7. 完整得分排行（前10，动量×flow叠加后）
    print()
    print("── 综合得分排行（前10，动量×flow）" + "─" * 10)
    top10 = scores.sort_values(ascending=False).head(10)
    for code, score in top10.items():
        name = names.get(code, code)
        flag = "✓" if code in target else " "
        trend = "↑" if score > 0 else "↓"
        print(f"  {flag} {code}  {name:<16}  {trend} {score:+.3f}")

    # 8. 保存信号记录
    target_scores = pos_scores.nlargest(TOP_N) if target else pd.Series(dtype=float)
    save_signal(str(today.date()), target, target_scores)
    print(f"\n信号已记录：{SIGNAL_LOG}")


if __name__ == "__main__":
    main()
