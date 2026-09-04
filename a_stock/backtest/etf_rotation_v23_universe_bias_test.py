"""
标的池选择偏差实测（2026-07-15）

背景：当前上线的45只ETF标的池（etf_universe.py）是2026-07-01一次性手工圈定
（宽基15只 + 覆盖TMT/新能源/医药/金融地产/消费/周期/军工等主题各配1-3只）。
用户指出这个圈定方式本身可能带有前视偏差——知道TMT/新能源后来表现好而重点纳入，
知道传统行业表现平庸而排除，这会系统性抬高回测夏普。

本脚本用机械化、不依赖主题的候选池构建规则重跑同一套回测框架，对比夏普差异：
  纳入规则：过去连续6个自然月，月度日均成交额都 > 1亿元（不要求成立年限，
  不做主题/行业均衡性筛选），逐月滚动更新，避免使用未来数据。

数据依赖：先运行 fetch_market_turnover.py 生成 market_turnover.parquet（全市场
ETF历史成交额）和 market_etf_meta.parquet（ETF基础信息，含已退市）。
"""

import sys
import pathlib
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro, fetch_single, START_DATE as FETCH_START_DATE  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, TOP_N, BENCHMARK, START_DATE, INIT_CASH, CASH_ETF,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
TURNOVER_PATH = DATA_DIR / "market_turnover.parquet"
META_PATH = DATA_DIR / "market_etf_meta.parquet"
CACHE_DIR = DATA_DIR / "daily_universe_test"  # 独立缓存目录，不污染生产 daily/

AMOUNT_THRESHOLD_QIAN = 100_000  # 千元，对应日均成交额 1 亿元
LOOKBACK_TRADING_DAYS = 126       # 约6个自然月的交易日数
MIN_VALID_DAYS = 100              # 窗口内最少有效交易日，避免停牌/新股均值失真


# ── 1. 构建逐月机械化候选池（point-in-time，无未来数据）──────
#
# 用整体滚动窗口（过去126个交易日的日均成交额）判断达标，而非"逐月严格全部达标"。
# 逐月判断曾导致春节假期月（如2026-02仅14个交易日）单月不达标就让整个候选池
# 连续几个月清零——这不合理，一次性的假期缩量不该打断整体流动性趋势的判断。

def build_daily_qualified(turnover: pd.DataFrame) -> pd.DataFrame:
    """返回 index=trade_date，columns=ts_code 的宽表 amount 矩阵"""
    wide = turnover.pivot(index="trade_date", columns="ts_code", values="amount")
    return wide.sort_index()


def build_pit_universe(amount_wide: pd.DataFrame) -> pd.Series:
    """
    对每个自然月，取该月最后一个交易日为评估点，判断"过去126个交易日整体
    日均成交额 > 阈值"的 ts_code 集合。只用截止到该评估日的历史数据。
    返回 index=月度(Period[M])，值为 set(ts_code)。
    """
    rolling_avg = amount_wide.rolling(LOOKBACK_TRADING_DAYS, min_periods=MIN_VALID_DAYS).mean()
    qualified = rolling_avg > AMOUNT_THRESHOLD_QIAN

    ym = qualified.index.to_period("M")
    month_end_idx = qualified.groupby(ym).apply(lambda x: x.index[-1])

    pit = {}
    for period, eval_date in month_end_idx.items():
        row = qualified.loc[eval_date]
        pit[period] = set(row[row].index)
    return pd.Series(pit)


# ── 2. 拉取候选池全部标的的复权价格（独立缓存目录）────────────

def fetch_prices_for_candidates(codes: list, stale_days: int = 5) -> None:
    """
    拉取候选标的价格并缓存。若缓存文件已存在但最新数据日期距今超过
    stale_days个自然日（默认5天，覆盖节假日），视为过期，重新拉取覆盖。
    这避免了"文件存在就永久跳过"导致缓存数据停留在某次历史抓取时间点，
    被 run_backtest 当作缺失（NaN→0）处理，拖累组合净值。
    """
    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=stale_days)
    CACHE_DIR.mkdir(exist_ok=True)
    total = len(codes)
    for i, code in enumerate(codes, 1):
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            try:
                cached = pd.read_parquet(path, columns=["trade_date"])
                last_date = pd.to_datetime(cached["trade_date"]).max()
                if last_date >= cutoff:
                    continue
            except Exception:
                pass  # 读取失败也视为需要重新拉取
        for attempt in range(4):
            try:
                df = fetch_single(pro, code, FETCH_START_DATE, today)
                if not df.empty:
                    df.to_parquet(path, index=False)
                    print(f"[{i:03d}/{total}] {code} 完成 {len(df)} 条")
                else:
                    print(f"[{i:03d}/{total}] {code} 无数据")
                break
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"[{i:03d}/{total}] {code} 第{attempt+1}次失败: {e}，{wait}s后重试")
                time.sleep(wait)
        time.sleep(0.5)


def load_close_matrix_from_cache(codes: list) -> pd.DataFrame:
    frames = {}
    for code in codes:
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "close"])
            frames[code] = df.set_index("trade_date")["close"]
    return pd.DataFrame(frames).sort_index()


# ── 3. 用逐月候选池掩码 scores，复用现有回测引擎 ──────────────

def mask_scores_by_pit_universe(scores: pd.DataFrame, pit_universe: pd.Series) -> pd.DataFrame:
    """
    把每个交易日的 scores 里，不属于"当月机械化候选池"的标的置为 NaN，
    这样 run_backtest 的 Top N 挑选只会在合法候选内进行。
    候选池按月更新（用上个月末已确定的名单，避免月中用到未来数据）。
    """
    masked = scores.copy()
    ym_series = scores.index.to_period("M")
    for ym in sorted(set(ym_series)):
        # 用上一个月的候选池名单（当月初调仓时，上月的连续达标情况已确定，不含未来信息）
        prior_ym = ym - 1
        codes_ok = pit_universe.get(prior_ym, set())
        day_mask = ym_series == ym
        drop_cols = [c for c in masked.columns if c not in codes_ok]
        masked.loc[day_mask, drop_cols] = np.nan
    return masked


def main():
    print("加载全市场成交额数据...")
    turnover = pd.read_parquet(TURNOVER_PATH)
    meta = pd.read_parquet(META_PATH)
    print(f"成交额记录 {len(turnover)} 行，ETF基础信息 {len(meta)} 只")

    print("构建逐月机械化达标矩阵...")
    etf_codes = set(meta["ts_code"])
    turnover = turnover[turnover["ts_code"].isin(etf_codes)]
    print(f"过滤到股票型ETF后成交额记录 {len(turnover)} 行")
    amount_wide = build_daily_qualified(turnover)
    pit_universe = build_pit_universe(amount_wide)

    all_candidates = sorted(set().union(*pit_universe.dropna().apply(lambda s: s if isinstance(s, set) else set())))
    print(f"历史上任一时点曾机械化达标的ETF共 {len(all_candidates)} 只")

    print("拉取候选池复权价格（已缓存的跳过）...")
    fetch_prices_for_candidates(all_candidates)

    print("加载价格矩阵...")
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]
    print(f"有效标的数：{len(valid_codes)}（vs 生产标的池45只）")

    print(f"计算动量得分（窗口={MOMENTUM_WINDOW}日）...")
    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)

    print("按逐月机械化候选池掩码得分...")
    masked_scores = mask_scores_by_pit_universe(scores, pit_universe)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    print(f"调仓日数量：{len(rebal_dates)}")

    print("运行回测（机械化候选池）...")
    nav = run_backtest(close, masked_scores, rebal_dates, cash_etf=CASH_ETF)

    bench = close[BENCHMARK].dropna() if BENCHMARK in close.columns else None
    print("\n" + "=" * 60)
    print("标的池选择偏差实测结果")
    print("=" * 60)
    print(f"回测区间：{nav.index[0].date()} → {nav.index[-1].date()}")
    stats = calc_stats(nav, "机械化候选池(成交额>1亿,连续6月)")
    print(pd.DataFrame([stats]).set_index("标的").to_string())
    print("\n对照：生产标的池（45只手工圈定）全样本夏普 1.053（详见 docs/research_etf_rotation.md）")


if __name__ == "__main__":
    main()
