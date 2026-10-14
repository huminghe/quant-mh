"""
补算 2026-10-08（10月首个交易日）本应产生但因 cron 权限问题漏跑的信号。

背景：signal_today.py 的 is_first_trading_day_of_month() 判断依赖
pd.Timestamp.today()，无法指定回填某个历史日期；本脚本复用其核心计算
逻辑（动量得分 momentum_score/get_scores_today、flow 打折
get_flow_boost_today 的算法部分），但把数据截断到 2026-10-08 当天，
还原"如果 10-08 当天脚本正常运行"本应产生的结果，保持与 signal_log.csv
历史记录相同的口径（均为当日真实收盘价计算，无前视偏差）。

只运行一次，运行后 backfill 过的记录即永久写入 signal_log.csv，无需重跑。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, run_update, init_pro

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all

from signal_today import (
    MOMENTUM_WINDOW, RISK_VOL_WINDOW, TOP_N, FLOW_LOOKBACK_MONTHS,
    CANDIDATES_FILE, BENCHMARK_FILE, SIGNAL_LOG,
    load_candidate_codes, load_etf_names, get_scores_today,
    load_last_holdings, save_signal,
)

BACKFILL_DATE = "2026-10-08"


def get_flow_boost_asof(codes: list, asof: str) -> pd.Series:
    """get_flow_boost_today 的回填版：份额数据截断到 asof 当天，不用最新数据"""
    pro = init_pro()
    start = (pd.Timestamp(asof) - pd.DateOffset(months=FLOW_LOOKBACK_MONTHS)).strftime("%Y%m%d")
    share_matrix = fetch_fund_share_all(pro, codes, start_date=start)
    if share_matrix.empty:
        return pd.Series(dtype=float)
    share_matrix = share_matrix.loc[:asof]
    monthly_share = share_matrix.resample("ME").last()
    flow_1m = monthly_share.pct_change()
    if flow_1m.empty:
        return pd.Series(dtype=float)
    latest = flow_1m.iloc[-1].dropna()
    if len(latest) < 5:
        return pd.Series(dtype=float)
    r = latest.rank(pct=True)
    return 1 - r


def main():
    candidates = load_candidate_codes()
    names = load_etf_names()

    print(f"更新行情数据（候选池 {len(candidates)} 只）...")
    run_update(codes=candidates)

    close_full = load_close_matrix(codes=candidates)
    close = close_full.loc[:BACKFILL_DATE]
    asof_date = close.index[-1]
    if str(asof_date.date()) != BACKFILL_DATE:
        print(f"警告：本地数据中 {BACKFILL_DATE} 当天无记录，实际取到最近一个交易日 {asof_date.date()}")
    print(f"回填数据日期：{asof_date.date()}\n")

    scores = get_scores_today(close, MOMENTUM_WINDOW, RISK_VOL_WINDOW)

    print("拉取flow信号（ETF份额月度变化率，截断到回填日期）...")
    boost = get_flow_boost_asof(list(scores.index), BACKFILL_DATE)
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

    target = list(pos_scores.nlargest(TOP_N).index)
    last_holdings = load_last_holdings()

    print("=" * 60)
    print(f"ETF 轮动月度信号（补算）  {asof_date.date()}")
    print("=" * 60)

    if not target:
        print("本月信号：全部空仓（无正动量标的）")
    else:
        print(f"本月目标持仓（等权，各约{100 // len(target)}%）：")
        for i, code in enumerate(target, 1):
            name = names.get(code, code)
            score = scores.get(code, 0)
            marker = " ← 新增" if code not in last_holdings else ""
            print(f"  {i}. {code}  {name:<16}  得分={score:.3f}{marker}")

    to_sell = [c for c in last_holdings if c not in target]
    to_buy = [c for c in target if c not in last_holdings]
    print()
    print(f"上月持仓：{', '.join(last_holdings) if last_holdings else '无记录'}")
    if to_sell:
        print("需要卖出：")
        for code in to_sell:
            print(f"  - {code}  {names.get(code, code)}")
    if to_buy:
        print("需要买入：")
        for code in to_buy:
            print(f"  + {code}  {names.get(code, code)}")

    print()
    print("── 综合得分排行（前10，动量×flow）" + "─" * 10)
    top10 = scores.sort_values(ascending=False).head(10)
    for code, score in top10.items():
        name = names.get(code, code)
        flag = "✓" if code in target else " "
        trend = "↑" if score > 0 else "↓"
        print(f"  {flag} {code}  {name:<16}  {trend} {score:+.3f}")

    target_scores = pos_scores.nlargest(TOP_N) if target else pd.Series(dtype=float)
    save_signal(str(asof_date.date()), target, target_scores)
    print(f"\n信号已补记：{SIGNAL_LOG}")


if __name__ == "__main__":
    main()
