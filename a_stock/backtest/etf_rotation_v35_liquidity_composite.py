"""
候选池构建规则优化 B-3：多重流动性代理综合排名（2026-07-27）

背景：v29-v33证伪"放宽成交额门槛扩容"，v34证伪"同跟踪指数去重"。用户要求测试
第三个新方向——不用单一维度（成交额）判断候选池准入，而是组合多个流动性代理
指标。调研发现学术上（Fong/Holden/Trzcinka 2017《Best Liquidity Proxies for
Global Research》）Amihud和Corwin-Schultz(High-Low Impact)在日频代理中并列
最优，且都能用现有OHLC价格缓存零成本计算，不需要新拉数据。

规则（与已证伪的"申万行业分层"/"同指数去重"不同——那两者都是"分组内砍到只留
1只"，丧失组内轮动空间；本规则是"综合排名砍尾部"，收窄幅度可控，不做强制分组）：
  1. 成交额门槛（现有规则）先筛出及格候选池
  2. 及格候选池内，按过去126日滚动 Amihud非流动性、Corwin-Schultz价差两个
     指标分别算百分位排名，取均值为综合流动性排名
  3. 剔除综合排名最差的后20%（drop_frac可调），剩余作为最终候选池

Amihud = |日收益率| / 成交额，值越低=流动性越好
Corwin-Schultz = 用连续两日最高价/最低价比值估计隐含买卖价差，值越低=流动性越好
两者捕捉的维度不同（一个是"冲击成本"，一个是"报价价差"），理论上比单一维度更全面。

数据依赖：fetch_market_turnover.py（market_turnover.parquet/market_etf_meta.parquet）
         价格缓存 daily_universe_test/*.parquet（已含 open/high/low/close/vol/amount）
"""

import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, START_DATE, CASH_ETF,
)
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    TURNOVER_PATH, META_PATH, CACHE_DIR, LOOKBACK_TRADING_DAYS, MIN_VALID_DAYS,
    AMOUNT_THRESHOLD_QIAN, build_daily_qualified, build_pit_universe,
    fetch_prices_for_candidates, load_close_matrix_from_cache, mask_scores_by_pit_universe,
)
from etf_rotation_v34_dedup_by_benchmark import split_in_out_sample  # noqa: E402

DROP_FRAC = 0.2  # 剔除综合流动性排名最差的后20%


# ── 1. 从价格缓存算 Amihud / Corwin-Schultz 两个流动性代理 ──────

def load_hl_matrix_from_cache(codes: list):
    """返回 (high_wide, low_wide)，index=trade_date，columns=ts_code"""
    high_frames, low_frames = {}, {}
    for code in codes:
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "high", "low"])
            df = df.set_index("trade_date")
            high_frames[code] = df["high"]
            low_frames[code] = df["low"]
    return pd.DataFrame(high_frames).sort_index(), pd.DataFrame(low_frames).sort_index()


def calc_amihud_daily(close_wide: pd.DataFrame, amount_wide: pd.DataFrame) -> pd.DataFrame:
    """|日收益率| / 成交额，值越低流动性越好"""
    ret = close_wide.pct_change()
    amount_aligned = amount_wide.reindex(index=close_wide.index, columns=close_wide.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = ret.abs() / amount_aligned
    return amihud.replace([np.inf, -np.inf], np.nan)


def calc_corwin_schultz_daily(high_wide: pd.DataFrame, low_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Corwin & Schultz (2012) 高低价差估计量，用连续两日(t, t+1)的最高/最低价
    估计隐含买卖价差，只需daily OHLC。负值截断为0（无法识别价差时视为0）。
    """
    ln_hl = np.log(high_wide / low_wide)
    beta = ln_hl ** 2 + ln_hl.shift(-1) ** 2

    h_next = high_wide.shift(-1)
    l_next = low_wide.shift(-1)
    h_max = np.maximum(high_wide, h_next)
    l_min = np.minimum(low_wide, l_next)
    gamma = np.log(h_max / l_min) ** 2

    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


# ── 2. 成交额门槛 + 综合流动性排名砍尾部 ────────────────────

def build_pit_universe_liquidity_rank(
    amount_wide: pd.DataFrame, amihud_daily: pd.DataFrame, cs_daily: pd.DataFrame,
    drop_frac: float = DROP_FRAC,
) -> pd.Series:
    """
    与 build_pit_universe 相同的成交额门槛逐月评估，叠加：及格候选内按
    Amihud/Corwin-Schultz 综合百分位排名剔除最差 drop_frac。缺少足够历史
    （滚动窗口不满min_periods）的标的保留原判断，不因数据不足被误伤。
    """
    rolling_avg = amount_wide.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()
    qualified = rolling_avg > AMOUNT_THRESHOLD_QIAN

    amihud_roll = amihud_daily.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()
    cs_roll = cs_daily.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()

    ym = qualified.index.to_period("M")
    month_end_idx = qualified.groupby(ym).apply(lambda x: x.index[-1])

    pit = {}
    for period, eval_date in month_end_idx.items():
        row = qualified.loc[eval_date]
        codes = set(row[row].index)

        if eval_date not in amihud_roll.index or eval_date not in cs_roll.index:
            pit[period] = codes
            continue

        codes_list = list(codes)
        amihud_vals = amihud_roll.loc[eval_date].reindex(codes_list)
        cs_vals = cs_roll.loc[eval_date].reindex(codes_list)
        valid_mask = amihud_vals.notna() & cs_vals.notna()

        codes_valid = amihud_vals[valid_mask].index
        codes_novalid = codes - set(codes_valid)  # 历史不足，无法排名，保留

        if len(codes_valid) == 0:
            pit[period] = codes
            continue

        amihud_pct = amihud_vals[codes_valid].rank(pct=True)
        cs_pct = cs_vals[codes_valid].rank(pct=True)
        composite = (amihud_pct + cs_pct) / 2
        cutoff = composite.quantile(1 - drop_frac)
        kept_valid = set(composite[composite <= cutoff].index)

        pit[period] = kept_valid | codes_novalid
    return pd.Series(pit)


def run_one(label: str, pit_universe: pd.Series, all_codes: list):
    fetch_prices_for_candidates(all_codes)
    close_full = load_close_matrix_from_cache(all_codes)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]

    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]

    nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF)

    stats_full = calc_stats(nav, f"{label}(候选{len(valid_codes)}只,全样本)")
    nav_in, nav_out = split_in_out_sample(nav, frac=0.8)
    stats_in = calc_stats(nav_in, f"{label}(样本内80%)")
    stats_out = calc_stats(nav_out, f"{label}(样本外20%)")

    return len(valid_codes), stats_full, stats_in, stats_out


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    amount_wide = build_daily_qualified(turnover)

    print("构建基线候选池（v23原始规则，仅成交额门槛）...")
    pit_baseline = build_pit_universe(amount_wide)
    all_baseline = sorted(set().union(
        *pit_baseline.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))
    print(f"基线候选池历史累计标的数：{len(all_baseline)}")

    print("拉取/确认候选价格缓存（含OHLC，用于流动性代理计算）...")
    fetch_prices_for_candidates(all_baseline)

    print("加载OHLC矩阵，计算Amihud与Corwin-Schultz流动性代理...")
    close_full = load_close_matrix_from_cache(all_baseline)
    high_full, low_full = load_hl_matrix_from_cache(all_baseline)
    amihud_daily = calc_amihud_daily(close_full, amount_wide)
    cs_daily = calc_corwin_schultz_daily(high_full, low_full)

    print(f"构建综合流动性排名候选池（剔除最差{DROP_FRAC:.0%}）...")
    pit_composite = build_pit_universe_liquidity_rank(amount_wide, amihud_daily, cs_daily)
    all_composite = sorted(set().union(
        *pit_composite.dropna().apply(lambda s: s if isinstance(s, set) else set())
    ))
    print(f"综合排名候选池历史累计标的数：{len(all_composite)}")

    latest_period = pit_baseline.index.max()
    print(f"\n最新评估期({latest_period})：基线{len(pit_baseline[latest_period])}只 "
          f"→ 综合排名筛选后{len(pit_composite[latest_period])}只")

    results = {}
    for label, pit_universe, all_codes in [
        ("基线(仅成交额门槛)", pit_baseline, all_baseline),
        (f"综合流动性排名(砍尾{DROP_FRAC:.0%})", pit_composite, all_composite),
    ]:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        n_valid, stats_full, stats_in, stats_out = run_one(label, pit_universe, all_codes)
        results[label] = dict(n_valid=n_valid, full=stats_full, in_=stats_in, out=stats_out)
        for tag, s in [("全样本", stats_full), ("样本内80%", stats_in), ("样本外20%", stats_out)]:
            print(f"\n[{tag}]")
            for k, v in s.items():
                if k != "标的":
                    print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    rows = []
    for label, r in results.items():
        for tag, s in [("全样本", r["full"]), ("样本内80%", r["in_"]), ("样本外20%", r["out"])]:
            row = dict(s)
            row["配置"] = label
            row["区间"] = tag
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["配置", "区间"])
    print(df[["总收益", "年化收益(CAGR)", "年化夏普", "最大回撤", "年化波动率", "Calmar"]].to_string())

    for label, r in results.items():
        sharpe_in = float(r["in_"]["年化夏普"])
        sharpe_out = float(r["out"]["年化夏普"])
        ratio = sharpe_out / sharpe_in if sharpe_in != 0 else float("nan")
        flag = "（样本外<样本内*0.5，可能过拟合）" if sharpe_in > 0 and sharpe_out < sharpe_in * 0.5 else ""
        print(f"\n{label}：样本外/样本内夏普比 = {ratio:.2f}{flag}")


if __name__ == "__main__":
    main()
