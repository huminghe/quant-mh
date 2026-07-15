"""
ML弱信号集成 —— 影子监控（不替换线上signal_today.py评分）

背景：第十二轮调研（etf_rotation_v15_weak_signal_ensemble*.py）发现拥挤度+
成交量确认+资金流反向三信号等权集成，通过IC与稳健性检验（组合回测夏普
1.235 vs 基线1.053），但样本仅6.5年，建议先小规模试用观察，不直接替换
现有评分公式。本脚本与signal_today.py同期运行，额外计算"动量+集成信号
连续打折"版本的目标持仓，写入独立日志与线上版对比，观察1-2个季度后再
决定是否采用。详见 a_stock/docs/research.md 第十二轮小节。
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch_data import load_close_matrix, init_pro, run_update
from etf_universe import ETF_UNIVERSE
from signal_today import get_scores_today, MOMENTUM_WINDOW, RISK_VOL_WINDOW, TOP_N
from etf_rotation_v16_signal_combo_ablation import calc_crowding, load_amount_matrix, fetch_fund_share_all

SHADOW_LOG = pathlib.Path(__file__).parent / "results" / "shadow_signal_log.csv"


# ── 集成信号（复用第十二/十三轮已验证的计算逻辑）──────────────

def compute_ensemble_boost(close: pd.DataFrame, pro) -> pd.Series:
    """
    计算最新交易日的集成信号打分（拥挤度+成交量确认+资金流反向，横截面排名等权平均）。
    返回：Series，index=ts_code，值域[0,1]，越大越好；缺信号的标的按可用信号平均。
    """
    today = close.index[-1]

    print("  计算拥挤度信号（全历史滚动相关性，较慢）...")
    crowding = calc_crowding(close)
    crowd_today = crowding.iloc[-1]

    print("  计算成交量确认信号...")
    amount = load_amount_matrix().reindex(columns=close.columns)
    vol_ratio_today = (amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)).iloc[-1]

    print("  拉取近期ETF份额数据（资金流信号）...")
    start_flow = (today - pd.Timedelta(days=100)).strftime("%Y%m%d")
    share = fetch_fund_share_all(pro, list(close.columns), start_date=start_flow)
    monthly_share = share.resample("ME").last() if not share.empty else pd.DataFrame()
    flow_today = monthly_share.pct_change().iloc[-1] if len(monthly_share) >= 2 else pd.Series(dtype=float)

    ranks = []
    for s, invert in [(crowd_today, True), (vol_ratio_today, False), (flow_today, True)]:
        s = s.dropna()
        if len(s) < 5:
            continue
        r = s.rank(pct=True)
        if invert:
            r = 1 - r
        ranks.append(r)

    if not ranks:
        return pd.Series(dtype=float)
    return pd.concat(ranks, axis=1).mean(axis=1)


# ── 日志 ──────────────────────────────────────────────────

def save_shadow_signal(date: str, online_target: list, ensemble_target: list) -> None:
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "日期": date,
        "线上持仓1": online_target[0] if len(online_target) > 0 else "现金",
        "线上持仓2": online_target[1] if len(online_target) > 1 else "现金",
        "线上持仓3": online_target[2] if len(online_target) > 2 else "现金",
        "集成持仓1": ensemble_target[0] if len(ensemble_target) > 0 else "现金",
        "集成持仓2": ensemble_target[1] if len(ensemble_target) > 1 else "现金",
        "集成持仓3": ensemble_target[2] if len(ensemble_target) > 2 else "现金",
        "持仓是否一致": online_target == ensemble_target,
    }
    if SHADOW_LOG.exists():
        log = pd.read_csv(SHADOW_LOG)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(SHADOW_LOG, index=False)


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("更新行情数据...")
    run_update()

    close = load_close_matrix()
    today = close.index[-1]
    print(f"最新数据日期：{today.date()}\n")

    print("计算线上评分（纯风险调整动量）...")
    online_scores = get_scores_today(close, MOMENTUM_WINDOW, RISK_VOL_WINDOW)
    online_target = list(online_scores[online_scores > 0].nlargest(TOP_N).index)

    print("计算集成信号（拥挤度+成交量确认+资金流，连续打折叠加）...")
    pro = init_pro()
    boost = compute_ensemble_boost(close, pro)

    ensemble_scores = online_scores.copy()
    for code in ensemble_scores.index:
        if code in boost.index and not pd.isna(boost[code]):
            ensemble_scores[code] *= (0.5 + boost[code])
    ensemble_target = list(ensemble_scores[ensemble_scores > 0].nlargest(TOP_N).index)

    print("\n" + "=" * 70)
    print(f"ETF 轮动信号对比（影子监控）  {today.date()}")
    print("=" * 70)

    print(f"\n线上版（纯动量，实盘执行）：")
    for i, code in enumerate(online_target, 1):
        print(f"  {i}. {code}  {ETF_UNIVERSE.get(code, code)}  得分={online_scores[code]:.3f}")

    print(f"\n集成版（动量+弱信号打折，仅监控不执行）：")
    for i, code in enumerate(ensemble_target, 1):
        print(f"  {i}. {code}  {ETF_UNIVERSE.get(code, code)}  得分={ensemble_scores[code]:.3f}")

    if online_target == ensemble_target:
        print("\n本月两版持仓一致。")
    else:
        only_online = [c for c in online_target if c not in ensemble_target]
        only_ensemble = [c for c in ensemble_target if c not in online_target]
        print("\n本月两版持仓存在差异：")
        if only_online:
            print(f"  仅线上版持有：{', '.join(only_online)}")
        if only_ensemble:
            print(f"  仅集成版持有：{', '.join(only_ensemble)}")

    save_shadow_signal(str(today.date()), online_target, ensemble_target)
    print(f"\n影子信号已记录：{SHADOW_LOG}")


if __name__ == "__main__":
    main()
