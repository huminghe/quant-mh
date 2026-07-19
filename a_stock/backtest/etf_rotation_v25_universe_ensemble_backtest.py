"""
机械化候选池 + ML弱信号集成 交叉验证（2026-07-17）

背景：两个此前独立验证过的维度从未交叉测试——
  ① 标的池选择偏差（v23）：45只手工圈定标的池夏普1.053 vs 机械化候选池
     （成交额>1亿元连续6月达标，无数量上限，历史峰值431只）夏普0.59。
  ② ML弱信号集成（v15/v16）：45只手工标的池上，纯动量基线1.053，
     叠加拥挤度+成交量确认+资金流三信号集成后提升至1.235。

本脚本回答：把①的机械化候选池 + ②的三信号集成方案组合起来，历史表现
是否优于机械化候选池纯动量（0.59）？能否部分弥补标的池选择偏差？
还是集成信号的价值本身就依赖于手工标的池已隐含的行业筛选，换到机械化
候选池上不再生效？

数据依赖：复用 v23 已缓存的 daily_universe_test/ 价格数据（含 amount 列，
无需重新拉取）。资金流份额数据（fund_share）需要对431只候选新拉取。
"""

import sys
import pathlib
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH,
    build_daily_qualified, build_pit_universe,
    load_close_matrix_from_cache, mask_scores_by_pit_universe,
    CACHE_DIR,
)
from etf_rotation import calc_all_scores, get_rebalance_dates, calc_stats  # noqa: E402
from etf_rotation_v16_signal_combo_ablation import (  # noqa: E402
    calc_crowding, fetch_fund_share_all, run_backtest as run_backtest_v16,
)

START_DATE = "2016-01-01"
MOMENTUM_WINDOW = 25

# 已知结果，仅作对照展示，不重跑
KNOWN_MANUAL_POOL_BASELINE = 1.053   # 45只手工标的池，纯动量，全样本
KNOWN_MANUAL_POOL_ENSEMBLE = 1.235   # 45只手工标的池，+三信号集成
KNOWN_MECH_POOL_BASELINE = 0.59      # 机械化候选池，纯动量，全样本（v23）


def load_amount_matrix_from_cache(codes: list) -> pd.DataFrame:
    """从 v23 缓存的价格 parquet 里额外读取 amount 列（同一批文件，无需重新拉取）"""
    frames = {}
    for code in codes:
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "amount"])
            frames[code] = df.set_index("trade_date")["amount"]
    return pd.DataFrame(frames).sort_index()


def combo_rank_ensemble(crowding: pd.DataFrame, vol_ratio: pd.DataFrame,
                         flow_1m: pd.DataFrame, rebal_dates: list) -> pd.DataFrame:
    """三信号（拥挤度/成交量确认/资金流）按月横截面排名等权集成，逻辑与v16一致"""
    signal_ranks = {"crowding": {}, "vol_ratio": {}, "flow": {}}
    for d in rebal_dates:
        crowd_d = crowding.loc[d] if d in crowding.index else pd.Series(dtype=float)
        volr_d = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)
        flow_d = pd.Series(dtype=float)
        if not flow_1m.empty:
            idx = flow_1m.index[flow_1m.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m.loc[idx[-1]]

        for name, s, invert in [("crowding", crowd_d, True), ("vol_ratio", volr_d, False), ("flow", flow_d, True)]:
            s = s.dropna()
            if len(s) < 5:
                continue
            r = s.rank(pct=True)
            if invert:
                r = 1 - r
            signal_ranks[name][d] = r

    rank_dfs = {name: pd.DataFrame(d).T for name, d in signal_ranks.items()}
    parts = [df for df in rank_dfs.values() if not df.empty]
    if not parts:
        return pd.DataFrame()
    common_dates = parts[0].index
    for other in parts[1:]:
        common_dates = common_dates.union(other.index)
    combined = {}
    for d in common_dates:
        vals = [p.loc[d] for p in parts if d in p.index]
        if vals:
            combined[d] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(combined).T


def main():
    print("加载全市场成交额数据，构建机械化候选池（复用v23逻辑）...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    amount_wide = build_daily_qualified(turnover)
    pit_universe = build_pit_universe(amount_wide)
    all_candidates = sorted(set().union(*pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())))
    print(f"历史上任一时点曾机械化达标的ETF共 {len(all_candidates)} 只")

    print("加载价格矩阵（复用v23缓存）...")
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]
    print(f"有效标的数：{len(valid_codes)}")

    print(f"计算动量得分（窗口={MOMENTUM_WINDOW}日）...")
    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    print(f"调仓日数量：{len(rebal_dates)}")

    print("重跑机械化候选池纯动量基线（校验与v23一致）...")
    nav_base = run_backtest_v16(close, masked_scores, rebal_dates)
    stats_base = calc_stats(nav_base, "机械化候选池·纯动量")
    sharpe_base = float(stats_base["年化夏普"])

    print("计算拥挤度信号（431只标的两两相关性，较慢）...")
    crowding = calc_crowding(close)

    print("计算成交量确认信号（复用v23缓存里的amount列）...")
    amount = load_amount_matrix_from_cache(valid_codes)
    amount = amount[[c for c in valid_codes if c in amount.columns]]
    amount_na_ratio = amount.isna().mean().mean()
    print(f"  amount数据缺失率：{amount_na_ratio:.1%}")
    vol_ratio = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)

    print("拉取431只候选的资金流份额数据（fund_share，预计1-2分钟）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()
    print(f"  资金流数据覆盖：{share_matrix.shape[1] if not share_matrix.empty else 0} 只标的")

    print("三信号按月横截面排名等权集成...")
    boost = combo_rank_ensemble(crowding, vol_ratio, flow_1m, rebal_dates)

    print("运行回测（机械化候选池 + 三信号集成，连续打折叠加）...")
    nav_ensemble = run_backtest_v16(close, masked_scores, rebal_dates, boost_signal=boost, boost_mode="continuous")
    stats_ensemble = calc_stats(nav_ensemble, "机械化候选池·+三信号集成")
    sharpe_ensemble = float(stats_ensemble["年化夏普"])

    print("\n" + "=" * 70)
    print("机械化候选池 + ML弱信号集成 交叉验证结果")
    print("=" * 70)
    print(f"回测区间：{nav_base.index[0].date()} → {nav_base.index[-1].date()}")

    rows = [
        {"方案": "45只手工标的池·纯动量（已知，未重跑）", "夏普": KNOWN_MANUAL_POOL_BASELINE},
        {"方案": "45只手工标的池·+三信号集成（已知，未重跑）", "夏普": KNOWN_MANUAL_POOL_ENSEMBLE},
        {"方案": "机械化候选池·纯动量（v23已知，本次重跑校验）", "夏普": sharpe_base},
        {"方案": "机械化候选池·+三信号集成（本次新算）", "夏普": sharpe_ensemble},
    ]
    df = pd.DataFrame(rows).set_index("方案")
    df["夏普"] = df["夏普"].map(lambda x: f"{x:.3f}")
    print(df.to_string())

    delta = sharpe_ensemble - sharpe_base
    print(f"\n机械化候选池上，集成信号带来的Δ夏普 = {delta:+.3f}")
    print(f"（对照：同样的三信号集成，在45只手工标的池上Δ夏普 = {KNOWN_MANUAL_POOL_ENSEMBLE - KNOWN_MANUAL_POOL_BASELINE:+.3f}）")

    print("\n详细统计（机械化候选池两个方案）：")
    detail = pd.DataFrame([stats_base, stats_ensemble]).set_index("标的")
    print(detail.to_string())


if __name__ == "__main__":
    main()
