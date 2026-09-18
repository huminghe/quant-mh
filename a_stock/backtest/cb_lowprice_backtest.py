"""
可转债双低策略：信用过滤叠加 + 完整组合回测（Phase 2 + Phase 3）

承接 factor_ic_cb_lowprice.py 的Phase 1结论，做两件事：
1. 信用过滤叠加（Phase 2）：用PIT安全的正股ST状态 + 资产负债率过滤双低池，
   对比"双低+过滤" vs "双低不过滤"两个版本，重点看2023年前后违约密集期
2. 完整组合回测+验收（Phase 3）：按 .claude/rules/trading-standards.md
   验收标准，样本内外拆分（以2022-01-01新规为分段点）输出年化收益/夏普/
   最大回撤/胜率盈亏比

信用过滤设计（PIT安全）：
- ST状态：用 cb_stk_namechange.parquet（tushare namechange历史区间）判断
  月末时点正股名称是否含'ST'字样，不用当前快照（避免前视偏差）
- 资产负债率：用 financials/{stk_code}.parquet 按 ann_date 做PIT查询
  （fetch_financials.py已有全市场历史数据），剔除资产负债率高于80%分位
  的正股对应转债（高杠杆是违约风险的直接代理指标）
- 转债本身评级不用（akshare接口只返回当前快照，是前视偏差，Phase 0已确认）

成本模型：可转债无印花税，只有佣金+滑点。按trading-standards.md：
佣金万1双边（买卖各收）=0.02%，滑点万2双边=0.04%，回合成本=0.06%
（对比股票ROUND_TRIP_COST=0.164%，主要差异是无千1印花税）

用法：
  cd a_stock/backtest
  python cb_lowprice_backtest.py
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
UNIVERSE_FILE = DATA_DIR / "cb_universe.parquet"
PREMIUM_FILE = DATA_DIR / "cb_premium_history.parquet"
NAMECHANGE_FILE = DATA_DIR / "cb_stk_namechange.parquet"
FINANCIALS_DIR = DATA_DIR / "financials"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "cb_lowprice_backtest"

START_DATE = "2018-01-01"
END_DATE = "2026-08-31"
NEW_RULE_DATE = pd.Timestamp("2022-01-01")

MIN_BONDS_PER_CROSS = 30
TOP_PCT = 0.2
DEBT_TO_ASSETS_PCT_THRESHOLD = 0.8  # 剔除资产负债率高于80分位的正股对应转债

RISK_FREE_ANNUAL = 0.02
ROUND_TRIP_COST = 0.0006  # 可转债：佣金万1双边0.02% + 滑点万2双边0.04%，无印花税


# ── 数据加载 ──────────────────────────────────────────────

def load_premium_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    ph = pd.read_parquet(PREMIUM_FILE)
    ph = ph.dropna(subset=["close", "conv_premium_pct"]).copy()
    ph["low_score"] = ph["close"] + 100 * ph["conv_premium_pct"]
    close_panel = ph.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    low_panel = ph.pivot(index="trade_date", columns="ts_code", values="low_score").sort_index()
    return close_panel, low_panel


def load_cb_to_stk_map() -> dict[str, str]:
    cb = pd.read_parquet(UNIVERSE_FILE)
    return cb.dropna(subset=["stk_code"]).set_index("ts_code")["stk_code"].to_dict()


_fina_cache: dict[str, pd.DataFrame] = {}


def get_fina(stk_code: str) -> pd.DataFrame:
    if stk_code not in _fina_cache:
        path = FINANCIALS_DIR / f"{stk_code}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["ann_date"] = pd.to_datetime(df["ann_date"])
            _fina_cache[stk_code] = df.sort_values("ann_date").reset_index(drop=True)
        else:
            _fina_cache[stk_code] = pd.DataFrame()
    return _fina_cache[stk_code]


def get_debt_to_assets_pit(stk_code: str, as_of: pd.Timestamp) -> float | None:
    df = get_fina(stk_code)
    if df.empty or "debt_to_assets" not in df.columns:
        return None
    valid = df[(df["ann_date"] <= as_of) & df["debt_to_assets"].notna()]
    if valid.empty:
        return None
    return float(valid.iloc[-1]["debt_to_assets"])


def load_namechange_intervals() -> pd.DataFrame:
    df = pd.read_parquet(NAMECHANGE_FILE)
    df["end_date"] = df["end_date"].fillna(pd.Timestamp("2099-12-31"))
    return df


def is_st_pit(stk_code: str, as_of: pd.Timestamp, nc_df: pd.DataFrame) -> bool:
    """PIT判断：as_of时点该正股名称是否含ST字样（用namechange历史区间，不用当前快照）"""
    rows = nc_df[nc_df["ts_code"] == stk_code]
    hit = rows[(rows["start_date"] <= as_of) & (as_of < rows["end_date"])]
    if hit.empty:
        return False
    return "ST" in str(hit.iloc[-1]["name"])


# ── 因子 + 过滤 + 分组 ────────────────────────────────────

def get_month_ends(close_panel: pd.DataFrame) -> np.ndarray:
    sub = close_panel.loc[START_DATE:END_DATE]
    nat_month_ends = sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        sub.index[sub.index <= m][-1]
        for m in nat_month_ends
        if len(sub.index[sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return monthly_last


def compute_debt_to_assets_cross(codes: list[str], cb_to_stk: dict, month_end: pd.Timestamp) -> pd.Series:
    vals = {}
    for code in codes:
        stk = cb_to_stk.get(code)
        if stk is None:
            continue
        v = get_debt_to_assets_pit(stk, month_end)
        if v is not None:
            vals[code] = v
    return pd.Series(vals)


def apply_credit_filter(
    codes: list[str], month_end: pd.Timestamp, cb_to_stk: dict, nc_df: pd.DataFrame
) -> list[str]:
    """PIT信用过滤：剔除正股ST + 资产负债率高于80分位的转债"""
    debt_cross = compute_debt_to_assets_cross(codes, cb_to_stk, month_end)
    if debt_cross.empty:
        threshold = np.inf
    else:
        threshold = debt_cross.quantile(DEBT_TO_ASSETS_PCT_THRESHOLD)

    kept = []
    for code in codes:
        stk = cb_to_stk.get(code)
        if stk is not None and is_st_pit(stk, month_end, nc_df):
            continue
        debt = debt_cross.get(code)
        if debt is not None and debt > threshold:
            continue
        kept.append(code)
    return kept


def compute_monthly_returns(
    close_panel: pd.DataFrame, low_panel: pd.DataFrame,
    cb_to_stk: dict, nc_df: pd.DataFrame, use_filter: bool
) -> pd.DataFrame:
    monthly_last = get_month_ends(close_panel)
    records = []

    for i, month_end in enumerate(monthly_last[:-1]):
        month_end = pd.Timestamp(month_end)
        next_end = pd.Timestamp(monthly_last[i + 1])

        close_row = close_panel.loc[month_end].dropna()
        fwd_row = close_panel.loc[next_end].dropna()
        common_ret = close_row.index.intersection(fwd_row.index)
        if len(common_ret) < MIN_BONDS_PER_CROSS:
            continue
        fwd_ret = fwd_row[common_ret] / close_row[common_ret] - 1
        fwd_ret = fwd_ret[fwd_ret.abs() < 1.0]

        if month_end not in low_panel.index:
            continue
        factor = low_panel.loc[month_end].dropna()
        available = factor.index.intersection(fwd_ret.index).tolist()
        if len(available) < MIN_BONDS_PER_CROSS:
            continue

        if use_filter:
            available = apply_credit_filter(available, month_end, cb_to_stk, nc_df)
            if len(available) < MIN_BONDS_PER_CROSS:
                continue

        n_select = max(1, int(len(available) * TOP_PCT))
        selected = factor[available].nsmallest(n_select).index
        gross_ret = fwd_ret[selected].mean()
        turnover_est = 0.5  # 月度调仓，Top20%池部分重叠，估计单边50%换手
        cost = turnover_est * ROUND_TRIP_COST
        net_ret = gross_ret - cost
        bench_ret = fwd_ret[available].mean()

        records.append({
            "date": month_end, "gross_ret": gross_ret, "strategy": net_ret,
            "benchmark": bench_ret, "n_pool": len(available), "n_selected": len(selected),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计（复用 factor_multi_backtest_v2.py 的函数定义） ──

def sharpe(ret: pd.Series, freq: int = 12) -> float:
    if len(ret) < 2:
        return np.nan
    ann = ret.mean() * freq
    std = ret.std() * np.sqrt(freq)
    return np.nan if std < 1e-8 else (ann - RISK_FREE_ANNUAL) / std


def max_drawdown(nav: pd.Series) -> float:
    return ((nav - nav.cummax()) / nav.cummax()).min()


def annual_return(nav: pd.Series, freq: int = 12) -> float:
    n = len(nav)
    if n < 2:
        return np.nan
    return (1 + nav.iloc[-1] / nav.iloc[0] - 1) ** (1 / (n / freq)) - 1


def win_rate_and_pl_ratio(ret: pd.Series) -> tuple[float, float]:
    """月胜率 + 盈亏比（平均盈利月收益 / 平均亏损月收益绝对值）"""
    ret = ret.dropna()
    if ret.empty:
        return np.nan, np.nan
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = len(wins) / len(ret)
    pl_ratio = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and len(wins) > 0 else np.nan
    return win_rate, pl_ratio


def report_stats(ret_df: pd.DataFrame, label: str) -> dict:
    strat = ret_df["strategy"].dropna()
    if strat.empty:
        print(f"\n  [{label}] 无有效数据")
        return {}
    nav = (1 + strat).cumprod()
    win_rate, pl_ratio = win_rate_and_pl_ratio(strat)
    stats = {
        "样本月数": len(strat),
        "年化收益": annual_return(nav),
        "年化夏普(rf=2%)": sharpe(strat),
        "最大回撤": max_drawdown(nav),
        "月胜率": win_rate,
        "盈亏比": pl_ratio,
    }
    print(f"\n  [{label}]（n={stats['样本月数']}月）")
    print(f"    年化收益={stats['年化收益']*100:+.2f}%  年化夏普={stats['年化夏普(rf=2%)']:+.3f}  "
          f"最大回撤={stats['最大回撤']*100:.2f}%  月胜率={stats['月胜率']*100:.1f}%  盈亏比={stats['盈亏比']:.3f}")
    return stats


def default_crisis_hit_rate(ret_df: pd.DataFrame, start: str, end: str) -> float:
    """2023年前后违约密集期：Top20%组月均超额是否为负（踩雷代理指标）"""
    window = ret_df.loc[start:end]
    excess = (window["gross_ret"] - window["benchmark"]).dropna()
    if excess.empty:
        return np.nan
    return (excess < 0).mean()


SPLIT_MIN_MONTHS = 18  # 样本内/外每段至少18个月，否则夏普估计噪音过大


def split_point_sensitivity(ret_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """样本内外分段点敏感性检验：Phase 3只用2022-01-01这一个分段点算样本
    内外夏普比，本函数把分段点从早到晚逐季度扫一遍，看"样本外夏普腰斩/
    可能过拟合"这个判定结论是否只是2022-01-01这一个特定切点的偶然产物，
    还是在大多数切点下都成立（绝对收益和超额收益两个口径都算）"""
    strat = ret_df["strategy"].dropna()
    excess = (ret_df["strategy"] - ret_df["benchmark"]).dropna()
    if strat.empty:
        return pd.DataFrame()

    candidates = pd.date_range(strat.index.min(), strat.index.max(), freq="QS")
    records = []
    for split in candidates:
        is_mask = strat.index < split
        oos_mask = strat.index >= split
        if is_mask.sum() < SPLIT_MIN_MONTHS or oos_mask.sum() < SPLIT_MIN_MONTHS:
            continue

        is_sharpe_abs = sharpe(strat[is_mask])
        oos_sharpe_abs = sharpe(strat[oos_mask])
        is_sharpe_excess = sharpe(excess[is_mask])
        oos_sharpe_excess = sharpe(excess[oos_mask])

        ratio_abs = (oos_sharpe_abs / is_sharpe_abs) if is_sharpe_abs and not np.isnan(is_sharpe_abs) and is_sharpe_abs != 0 else np.nan
        ratio_excess = (oos_sharpe_excess / is_sharpe_excess) if is_sharpe_excess and not np.isnan(is_sharpe_excess) and is_sharpe_excess != 0 else np.nan

        records.append({
            "split_date": split, "is_月数": int(is_mask.sum()), "oos_月数": int(oos_mask.sum()),
            "IS夏普(绝对)": is_sharpe_abs, "OOS夏普(绝对)": oos_sharpe_abs, "夏普比(绝对)": ratio_abs,
            "IS夏普(超额)": is_sharpe_excess, "OOS夏普(超额)": oos_sharpe_excess, "夏普比(超额)": ratio_excess,
        })

    df = pd.DataFrame(records).set_index("split_date")
    if df.empty:
        print(f"\n  [{label}] 无满足最小月数约束的分段点，无法检验")
        return df

    print(f"\n  [{label}] 分段点敏感性检验（n={len(df)}个候选分段点，每季度一个，每段≥{SPLIT_MIN_MONTHS}月）：")
    overfit_abs = (df["夏普比(绝对)"] < 0.5) & (df["IS夏普(绝对)"] > 0)
    overfit_excess = (df["夏普比(超额)"] < 0.5) & (df["IS夏普(超额)"] > 0)
    print(f"    绝对收益：夏普比<0.5(可能过拟合)的分段点占比={overfit_abs.mean():.1%}，"
          f"夏普比均值={df['夏普比(绝对)'].mean():.2f}  std={df['夏普比(绝对)'].std():.2f}")
    print(f"    超额收益：夏普比<0.5(可能过拟合)的分段点占比={overfit_excess.mean():.1%}，"
          f"夏普比均值={df['夏普比(超额)'].mean():.2f}  std={df['夏普比(超额)'].std():.2f}")
    return df


ROLLING_WINDOW_MONTHS = 24


def _rolling_stats_from_series(ret: pd.Series, label: str) -> pd.DataFrame:
    """滚动24个月窗口（每月滑动1次）年化收益/夏普/回撤的通用计算，供绝对收益
    和超额收益两个版本共用（避免重复实现同一段滚动窗口逻辑）"""
    ret = ret.dropna()
    records = []
    for i in range(ROLLING_WINDOW_MONTHS, len(ret) + 1):
        window = ret.iloc[i - ROLLING_WINDOW_MONTHS: i]
        nav = (1 + window).cumprod()
        records.append({
            "window_end": window.index[-1],
            "年化收益": annual_return(nav),
            "年化夏普(rf=2%)": sharpe(window),
            "最大回撤": max_drawdown(nav),
        })
    df = pd.DataFrame(records).set_index("window_end")
    if df.empty:
        print(f"\n  [{label}] 窗口数不足{ROLLING_WINDOW_MONTHS}月，无法计算滚动窗口")
        return df

    print(f"\n  [{label}] 滚动{ROLLING_WINDOW_MONTHS}月窗口（n={len(df)}个窗口）：")
    for col in ["年化收益", "年化夏普(rf=2%)", "最大回撤"]:
        s = df[col]
        unit = "%" if col != "年化夏普(rf=2%)" else ""
        scale = 100 if unit == "%" else 1
        print(f"    {col}：均值={s.mean()*scale:+.2f}{unit}  std={s.std()*scale:.2f}{unit}  "
              f"最小={s.min()*scale:+.2f}{unit}  最大={s.max()*scale:+.2f}{unit}")
    neg_sharpe_ratio = (df["年化夏普(rf=2%)"] < 0).mean()
    print(f"    窗口夏普<0占比={neg_sharpe_ratio:.1%}（越高说明结论对样本窗口越敏感）")
    return df


def rolling_window_stats(ret_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """滚动窗口检验（绝对收益版）：用strategy列（策略扣成本后的净收益），
    衡量策略本身涨跌，会混入可转债市场整体的beta周期性（新规前市场好、
    2022-2024转债走弱、2025年起回暖），不能单独说明选股能力（alpha）
    是否稳健——需搭配 rolling_window_excess_stats 一起看"""
    return _rolling_stats_from_series(ret_df["strategy"], label)


def rolling_window_excess_stats(ret_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """滚动窗口检验（超额收益版）：用strategy - benchmark（策略净收益减去
    当月全部可用转债池等权平均收益），剔除市场整体行情（beta）后单独衡量
    双低因子的选股能力（alpha）本身是否稳健。benchmark是全池被动持有的
    对照组毛收益，不需要额外扣策略专属换手成本，直接相减即可"""
    excess = (ret_df["strategy"] - ret_df["benchmark"]).rename("strategy")
    return _rolling_stats_from_series(excess, label)


# ── 主流程 ────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    close_panel, low_panel = load_premium_panel()
    cb_to_stk = load_cb_to_stk_map()
    nc_df = load_namechange_intervals()
    print(f"因子面板：{close_panel.shape}，正股映射：{len(cb_to_stk)}只，名称变更记录：{len(nc_df)}条")

    results = {}
    for use_filter, key in [(False, "无过滤"), (True, "信用过滤")]:
        print(f"\n计算 {key} 版本月度收益...")
        df = compute_monthly_returns(close_panel, low_panel, cb_to_stk, nc_df, use_filter)
        results[key] = df
        df.to_csv(OUTPUT_DIR / f"monthly_ret_{key}.csv")

    # Phase 2：违约密集期踩雷对比（2023年前后，取搜特/正邦等实质违约事件密集区间）
    print(f"\n{'='*70}")
    print("Phase 2：违约密集期（2023-01 ~ 2024-06）踩雷率对比")
    print(f"{'='*70}")
    for key, df in results.items():
        rate = default_crisis_hit_rate(df, "2023-01-01", "2024-06-30")
        print(f"  {key}：Top20%组跑输基准月份占比 = {rate*100:.1f}%")

    # Phase 3：完整回测 + 验收（样本内/外以2022-01-01新规为分段点）
    print(f"\n{'='*70}")
    print("Phase 3：完整组合回测验收（样本内 2018-2021，样本外 2022-2026）")
    print(f"{'='*70}")
    summary_rows = {}
    for key, df in results.items():
        print(f"\n--- {key} ---")
        is_df = df[df.index < NEW_RULE_DATE]
        oos_df = df[df.index >= NEW_RULE_DATE]
        is_stats = report_stats(is_df, f"{key} 样本内 2018-2021")
        oos_stats = report_stats(oos_df, f"{key} 样本外 2022-2026")
        if is_stats and oos_stats and is_stats["年化夏普(rf=2%)"] not in (0, np.nan):
            ratio = oos_stats["年化夏普(rf=2%)"] / is_stats["年化夏普(rf=2%)"] if is_stats["年化夏普(rf=2%)"] != 0 else np.nan
            flag = "可能过拟合" if (not np.isnan(ratio) and ratio < 0.5 and is_stats["年化夏普(rf=2%)"] > 0) else ""
            print(f"    样本内外夏普比 = {ratio:.2f}  {flag}")
        summary_rows[f"{key}_样本内"] = is_stats
        summary_rows[f"{key}_样本外"] = oos_stats

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.to_csv(OUTPUT_DIR / "is_oos_summary.csv")

    # Phase 4：滚动窗口稳健性检验（检验结论对2022-01-01单一分段点的敏感度）
    print(f"\n{'='*70}")
    print(f"Phase 4：滚动{ROLLING_WINDOW_MONTHS}月窗口稳健性检验（全样本2018-2026，逐月滑动，绝对收益）")
    print(f"{'='*70}")
    for key, df in results.items():
        rolling_df = rolling_window_stats(df, key)
        if not rolling_df.empty:
            rolling_df.to_csv(OUTPUT_DIR / f"rolling_{ROLLING_WINDOW_MONTHS}m_{key}.csv")

    # Phase 4b：同一滚动窗口检验，改用超额收益（strategy - benchmark），
    # 剔除可转债市场整体beta周期性，单独看双低因子选股能力（alpha）本身的稳健性
    print(f"\n{'='*70}")
    print(f"Phase 4b：滚动{ROLLING_WINDOW_MONTHS}月窗口稳健性检验（超额收益 strategy - benchmark）")
    print(f"{'='*70}")
    for key, df in results.items():
        rolling_excess_df = rolling_window_excess_stats(df, key)
        if not rolling_excess_df.empty:
            rolling_excess_df.to_csv(OUTPUT_DIR / f"rolling_{ROLLING_WINDOW_MONTHS}m_excess_{key}.csv")

    # Phase 5：分段点敏感性检验（Phase 3的"样本内外夏普比"判定是否只是
    # 2022-01-01这一个切点的偶然产物，逐季度扫描全部候选切点重新检验）
    print(f"\n{'='*70}")
    print(f"Phase 5：样本内外分段点敏感性检验（逐季度扫描全部候选切点）")
    print(f"{'='*70}")
    for key, df in results.items():
        split_df = split_point_sensitivity(df, key)
        if not split_df.empty:
            split_df.to_csv(OUTPUT_DIR / f"split_sensitivity_{key}.csv")

    print(f"\nn_trials说明：本轮只比较'双低默认参数不过滤' vs '双低+信用过滤'2个版本，"
          f"不做参数网格搜索，n_trials=2。")
    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
