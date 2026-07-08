"""
ETF轮动 + 行业内选股 混合策略回测（2026-07）

架构：
  Layer 1: ETF动量信号（风险调整动量 + 拥挤度软过滤 threshold=0.75, factor=0.2）→ Top3 ETF
  Layer 2: 行业ETF → 申万行业内中证500成分股 Top10
           宽基ETF → 直接持有ETF

验收标准：
  - 全样本超额（vs纯ETF轮动基线夏普1.005）> 0
  - IS超额 > 0
  - OOS超额 > -2%（允许略微下降，不能大幅恶化）
  - 2019-2021成长牛市期间不大幅跑输

用法：
  cd /path/to/quant-mh
  source venv/bin/activate
  python a_stock/backtest/etf_stock_hybrid_backtest.py
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "a_stock" / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from fetch_data import load_close_matrix
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR
from fetch_financials import load_financials
from etf_universe import BROAD_ETFS, SECTOR_ETFS

SW_INDUSTRY_FILE = DATA_DIR / "stock_sw_industry.parquet"


def get_sw_industry_map() -> dict:
    """申万一级行业映射（ts_code -> 行业名），与 ETF_TO_SECTOR 命名体系一致"""
    df = pd.read_parquet(SW_INDUSTRY_FILE)
    return df.set_index("ts_code")["sw_industry"].to_dict()

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH         = 1_000_000
ETF_COMMISSION    = 0.0001      # ETF佣金（双边各）
ETF_SLIPPAGE      = 0.0002      # ETF滑点（双边各）
STOCK_COMMISSION  = 0.0001      # 个股佣金（双边各）
STOCK_STAMP_DUTY  = 0.001       # 印花税（仅卖出）
STOCK_SLIPPAGE    = 0.0002      # 个股滑点（双边各）
RISK_FREE_ANNUAL  = 0.02

START_DATE        = "2016-01-01"
IS_END            = "2024-01-31"
OOS_START         = "2024-02-01"

MOMENTUM_WINDOW   = 25
RISK_VOL_WINDOW   = 21
TOP_N_ETF         = 3
TOP_N_STOCKS      = 10          # 每个行业ETF槽位选多少只个股
MIN_STOCKS_SECTOR = 3           # 行业内中证500成分股最少数量
MIN_HISTORY_QTRS  = 4           # 盈利增速稳定性最少季度数
MIN_STOCKS_CROSS  = 30          # 截面合成最少股票数

CORR_WINDOW       = 60
CORR_HIST_WINDOW  = 252
CROWDING_THRESHOLD = 0.75
CROWDING_FACTOR   = 0.2         # 软过滤系数（拥挤时动量得分 × 0.2）

HS500_MEMBERS_FILE = DATA_DIR / "hs500_members.parquet"

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── ETF → 申万行业 映射（设计文档定义） ──────────────────────
# 宽基ETF不在此表中，直接持有
ETF_TO_SECTOR = {
    "515000.SH": "计算机",
    "512760.SH": "电子",
    "159995.SZ": "电子",
    "515330.SH": "汽车",
    "516160.SH": "电力设备",
    "159629.SZ": "电力设备",
    "159596.SZ": "电力设备",
    "512010.SH": "医药生物",
    "512170.SH": "医药生物",
    "159992.SZ": "医药生物",
    "512800.SH": "银行",
    "512880.SH": "非银金融",
    "159931.SZ": "房地产",
    "512980.SH": "传媒",
    "159869.SZ": "传媒",
    "515030.SH": "电力设备",
    "159628.SZ": "机械设备",
    "516670.SH": "有色金属",
    "159975.SZ": "国防军工",
    "512660.SH": "国防军工",
    "512400.SH": "有色金属",
    "159928.SZ": "食品饮料",
    "515700.SH": "食品饮料",
    "159997.SZ": "食品饮料",
    "159801.SZ": "农林牧渔",
    "515220.SH": "基础化工",
    "159611.SZ": "煤炭",
    # 港股互联网/中概互联 → 视为宽基，直接持有
}

# 所有宽基ETF集合（直接持有，不做行业选股）
BROAD_ETF_CODES = set(BROAD_ETFS.keys())
# 港股/中概互联也视为宽基
BROAD_ETF_CODES |= {"516950.SH", "513050.SH", "510170.SH"}


# ── ICIR权重（来自factor_ic_quality_v2验证，中证500） ──────────
ICIR_WEIGHTS = {
    "profit_stability": 0.322,
    "ep_sector":        0.321,
    "ocf":              0.219,
    "roe":              0.195,
    "reversal":         0.123,
}

# ── 财务缓存 ──────────────────────────────────────────────

_fina_cache: dict[str, pd.DataFrame] = {}

def get_fina(ts_code: str) -> pd.DataFrame:
    if ts_code not in _fina_cache:
        _fina_cache[ts_code] = load_financials(ts_code)
    return _fina_cache[ts_code]


def get_fina_pit(ts_code: str, as_of: pd.Timestamp, field: str) -> float | None:
    df = get_fina(ts_code)
    if df.empty or field not in df.columns:
        return None
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    if valid.empty:
        return None
    return float(valid.iloc[-1][field])


def get_fina_history(ts_code: str, as_of: pd.Timestamp, field: str, n: int = 8) -> pd.Series:
    df = get_fina(ts_code)
    if df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    return valid[field].iloc[-n:].reset_index(drop=True)


# ── 因子计算（复用 factor_multi_backtest_v2 逻辑） ──────────

def compute_reversal(close_panel: pd.DataFrame, codes: list[str],
                     month_end: pd.Timestamp, window: int = 63) -> pd.Series:
    hist = close_panel[codes].loc[:month_end]
    if len(hist) < window + 1:
        return pd.Series(dtype=float)
    ret = hist.iloc[-1] / hist.iloc[-(window + 1)] - 1
    return -ret.dropna()


def compute_ep_sector(codes: list[str], month_end: pd.Timestamp,
                      close_row: pd.Series, industry_map: dict) -> pd.Series:
    ep_raw = {}
    for code in codes:
        eps = get_fina_pit(code, month_end, "eps")
        if eps is None or np.isnan(eps):
            continue
        price = close_row.get(code)
        if price is None or np.isnan(price) or price <= 0:
            continue
        ep_raw[code] = eps / price
    if not ep_raw:
        return pd.Series(dtype=float)

    ep_series = pd.Series(ep_raw)
    sector_series = pd.Series({c: industry_map.get(c, "未知") for c in ep_series.index})
    fallback_rank = ep_series.rank(pct=True)
    ranks = {}
    for code in ep_series.index:
        sector = sector_series[code]
        if sector == "未知":
            ranks[code] = fallback_rank[code]
            continue
        group = ep_series[sector_series == sector]
        if len(group) < 5:
            ranks[code] = fallback_rank[code]
        else:
            ranks[code] = group.rank(pct=True)[code]
    return pd.Series(ranks)


def compute_ocf(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "ocf_to_profit")
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


def compute_roe(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "roe_dt")
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


def compute_profit_stability(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        hist = get_fina_history(code, month_end, "netprofit_yoy", n=8)
        if len(hist) < MIN_HISTORY_QTRS:
            continue
        std = hist.std()
        if pd.notna(std) and std >= 0:
            values[code] = -std
    return pd.Series(values)


def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_composite_score(close_panel: pd.DataFrame, codes: list[str],
                             month_end: pd.Timestamp, industry_map: dict) -> pd.Series:
    """计算综合因子得分，返回 ts_code → score 的 Series（ICIR加权）"""
    close_row = close_panel[codes].loc[month_end].dropna() if month_end in close_panel.index else pd.Series()
    available = list(close_row.index) if not close_row.empty else codes

    raw_factors = {
        "reversal":         compute_reversal(close_panel, available, month_end),
        "ep_sector":        compute_ep_sector(available, month_end, close_row, industry_map),
        "ocf":              compute_ocf(available, month_end),
        "roe":              compute_roe(available, month_end),
        "profit_stability": compute_profit_stability(available, month_end),
    }

    norm = {}
    for fname, fs in raw_factors.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm[fname] = fs

    if not norm:
        return pd.Series(dtype=float)

    common = None
    for fs in norm.values():
        if common is None:
            common = set(fs.index)
        else:
            common &= set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return pd.Series(dtype=float)
    common = list(common)

    total_w = sum(ICIR_WEIGHTS.get(f, 0) for f in norm)
    if total_w < 1e-8:
        return pd.Series(dtype=float)

    score = pd.Series(0.0, index=common)
    for fname, fs in norm.items():
        w = ICIR_WEIGHTS.get(fname, 0) / total_w
        score += fs[common] * w
    return score.dropna()


# ── ETF动量得分计算（复用 etf_rotation_v3b_crowding 逻辑） ────

def momentum_score_single(prices: pd.Series) -> float:
    """OLS斜率 × R²，年化"""
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_etf_scores(close: pd.DataFrame) -> pd.DataFrame:
    """对每只ETF、每个交易日计算风险调整动量得分"""
    scores = {}
    for code in close.columns:
        series = close[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            pw = series.iloc[i - MOMENTUM_WINDOW: i]
            raw = momentum_score_single(pw)
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol = rets.std() * np.sqrt(252)
                raw = raw / vol if vol > 1e-6 else raw
            score_series.iloc[i] = raw
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close.index)


def calc_crowding(close: pd.DataFrame) -> pd.DataFrame:
    """每只ETF的拥挤度历史分位数（0~1）"""
    codes = list(close.columns)
    ret = close.pct_change()
    print(f"  计算 {len(codes)} 只ETF的拥挤度（窗口={CORR_WINDOW}日）...")

    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_WINDOW, len(close.index)):
        ret_win = ret.iloc[i - CORR_WINDOW: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(np.nanmean(corr_arr, axis=1), index=ret_win.columns)
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values

    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_HIST_WINDOW + CORR_WINDOW, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - CORR_HIST_WINDOW: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna()
            c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    # 每月第一个交易日作为调仓日（执行月末信号）
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def get_month_ends(index: pd.DatetimeIndex) -> list:
    """每月最后一个交易日（用于计算当月信号）"""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[-1]).tolist()


def apply_crowding(day_scores: pd.Series, date: pd.Timestamp,
                   crowding_pct: pd.DataFrame) -> pd.Series:
    """软过滤：拥挤度分位 > threshold 时得分 × CROWDING_FACTOR"""
    if date not in crowding_pct.index:
        return day_scores
    day_crowding = crowding_pct.loc[date]
    result = day_scores.copy()
    for code in result.index:
        if code not in day_crowding.index or pd.isna(day_crowding[code]):
            continue
        if day_crowding[code] > CROWDING_THRESHOLD:
            result[code] *= CROWDING_FACTOR
    return result


# ── 行业内选股（Layer 2） ─────────────────────────────────────

def select_sector_stocks(etf_code: str, month_end: pd.Timestamp,
                          stock_panel: pd.DataFrame, members_hs500: list[str],
                          industry_map: dict) -> list[str]:
    """
    给定行业ETF代码，从中证500成分股中筛选该申万行业内的Top10个股。
    返回股票代码列表，可能少于10只（成分股不足时取全部）。
    """
    target_sector = ETF_TO_SECTOR.get(etf_code)
    if target_sector is None:
        return []  # 映射表中不存在，降级为持有ETF

    # 找该行业内的中证500成分股
    sector_candidates = [
        c for c in members_hs500
        if industry_map.get(c) == target_sector and c in stock_panel.columns
    ]
    if len(sector_candidates) < MIN_STOCKS_SECTOR:
        # 行业内成分股太少，降级持有ETF
        return []

    # 计算综合因子得分，选Top N
    score = compute_composite_score(stock_panel, sector_candidates, month_end, industry_map)
    if score.empty:
        return []

    n = min(TOP_N_STOCKS, len(score))
    return score.nlargest(n).index.tolist()


# ── 月度收益计算（含成本） ────────────────────────────────────

def month_return_etf(code: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                     etf_close: pd.DataFrame) -> float | None:
    """计算单只ETF的月度净收益（扣成本）"""
    p0 = etf_close[code].get(entry_date) if code in etf_close.columns else None
    p1 = etf_close[code].get(exit_date) if code in etf_close.columns else None
    if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or p0 <= 0:
        return None
    gross = p1 / p0 - 1
    # 双边成本：买入+卖出
    cost = (ETF_COMMISSION + ETF_SLIPPAGE) * 2
    return gross - cost


def month_return_stocks(codes: list[str], entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                        stock_panel: pd.DataFrame) -> float | None:
    """计算一组个股的等权月度净收益（扣成本）"""
    rets = []
    for code in codes:
        p0 = stock_panel[code].get(entry_date) if code in stock_panel.columns else None
        p1 = stock_panel[code].get(exit_date) if code in stock_panel.columns else None
        if p0 is None or p1 is None or np.isnan(p0) or np.isnan(p1) or p0 <= 0:
            continue
        gross = p1 / p0 - 1
        # 个股成本：买入（佣金+滑点）+ 卖出（佣金+印花税+滑点）
        cost = (STOCK_COMMISSION + STOCK_SLIPPAGE) * 2 + STOCK_STAMP_DUTY
        rets.append(gross - cost)
    return np.mean(rets) if rets else None


# ── 主回测逻辑 ────────────────────────────────────────────────

def run_hybrid_backtest(
    etf_close: pd.DataFrame,
    etf_scores: pd.DataFrame,
    crowding_pct: pd.DataFrame,
    stock_panel: pd.DataFrame,
    hs500_members_file: pathlib.Path,
    industry_map: dict,
) -> pd.DataFrame:
    """
    月度混合策略回测。

    返回：records DataFrame，每行一个月，包含
      - date: 信号月末
      - strategy: 混合策略净收益
      - benchmark: 同期ETF轮动基线净收益
      - breakdown: 槽位明细（list）
    """
    month_ends = get_month_ends(etf_close.index)
    # 过滤到回测区间
    month_ends = [m for m in month_ends if m >= pd.Timestamp(START_DATE)]

    # 基线ETF轮动（仅ETF，无个股）：用于对比
    records = []

    for i, signal_date in enumerate(month_ends[:-1]):
        # 执行日 = 下个月第一个交易日（近似为下个月的第一个月末后的第一天）
        # 简化：用下个月月末的日期区间计算收益
        next_signal_date = month_ends[i + 1]

        # 找执行日（signal_date 之后第一个交易日）
        idx_pos = list(etf_close.index).index(signal_date) if signal_date in etf_close.index else -1
        if idx_pos < 0 or idx_pos + 1 >= len(etf_close.index):
            continue
        entry_date = etf_close.index[idx_pos + 1]  # 信号月末+1日买入
        exit_date = next_signal_date                # 下月末卖出

        # 获取ETF动量得分（用信号月末的得分）
        if signal_date not in etf_scores.index:
            continue
        day_scores = etf_scores.loc[signal_date].dropna().copy()

        # 拥挤度软过滤
        day_scores = apply_crowding(day_scores, signal_date, crowding_pct)

        # 选 Top3 正动量 ETF
        pos_scores = day_scores[day_scores > 0].nlargest(TOP_N_ETF)
        if pos_scores.empty:
            records.append({
                "date": signal_date,
                "strategy": 0.0,  # 空仓（无风险利率近似为0）
                "benchmark_etf_only": 0.0,
                "n_etf_slots": 0,
                "slots": [],
            })
            continue

        selected_etfs = list(pos_scores.index)

        # 基线：纯ETF轮动收益（等权）
        baseline_rets = []
        for etf in selected_etfs:
            r = month_return_etf(etf, entry_date, exit_date, etf_close)
            if r is not None:
                baseline_rets.append(r)
        baseline_ret = np.mean(baseline_rets) if baseline_rets else 0.0

        # PIT成分股（用信号月末）
        pit_members = load_members_pit(signal_date, members_file=hs500_members_file)

        # Layer 2：分槽位处理
        slot_rets = []
        slots_detail = []

        for etf_code in selected_etfs:
            if etf_code in BROAD_ETF_CODES or etf_code not in ETF_TO_SECTOR:
                # 宽基ETF或无映射 → 直接持有ETF
                r = month_return_etf(etf_code, entry_date, exit_date, etf_close)
                slot_type = "etf"
                if r is not None:
                    slot_rets.append(r)
                slots_detail.append({
                    "etf": etf_code,
                    "type": slot_type,
                    "stocks": [],
                    "ret": r,
                })
            else:
                # 行业ETF → 行业内选股
                stocks = select_sector_stocks(
                    etf_code, signal_date, stock_panel, pit_members, industry_map
                )
                if not stocks:
                    # 降级：行业成分股不足 → 持有ETF
                    r = month_return_etf(etf_code, entry_date, exit_date, etf_close)
                    slot_type = "etf_fallback"
                    if r is not None:
                        slot_rets.append(r)
                else:
                    r = month_return_stocks(stocks, entry_date, exit_date, stock_panel)
                    slot_type = "stocks"
                    if r is not None:
                        slot_rets.append(r)
                slots_detail.append({
                    "etf": etf_code,
                    "type": slot_type,
                    "stocks": stocks,
                    "ret": r,
                })

        hybrid_ret = np.mean(slot_rets) if slot_rets else 0.0

        records.append({
            "date":                signal_date,
            "strategy":            hybrid_ret,
            "benchmark_etf_only":  baseline_ret,
            "n_etf_slots":         len(selected_etfs),
            "slots":               slots_detail,
        })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("date")


# ── 绩效统计 ──────────────────────────────────────────────────

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


def print_stats(ret_df: pd.DataFrame, label: str, period: str = "全样本") -> None:
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark_etf_only"].dropna()
    common = strat.index.intersection(bench.index)
    strat, bench = strat[common], bench[common]
    excess = strat - bench

    nav_s = (1 + strat).cumprod()
    nav_b = (1 + bench).cumprod()

    print(f"\n  [{label} — {period}]")
    print(f"    样本月数: {len(strat)}")
    print(f"    混合策略年化: {annual_return(nav_s)*100:.1f}%")
    print(f"    ETF基线年化: {annual_return(nav_b)*100:.1f}%")
    print(f"    超额年化（vs ETF基线）: {excess.mean()*12*100:.1f}%")
    print(f"    混合策略夏普: {sharpe(strat):.3f}")
    print(f"    ETF基线夏普: {sharpe(bench):.3f}")
    print(f"    混合策略最大回撤: {max_drawdown(nav_s)*100:.1f}%")
    print(f"    月胜率（混合 vs ETF基线）: {(excess > 0).mean()*100:.1f}%")


def print_annual_excess(ret_df: pd.DataFrame) -> None:
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark_etf_only"].dropna()
    excess = (strat - bench).dropna()

    print("\n  年度超额（混合 vs ETF基线，月均）:")
    for y in sorted(excess.index.year.unique()):
        yr = excess[excess.index.year == y]
        print(f"    {y}: {yr.mean()*100:+.2f}%/月  (n={len(yr)})")


def plot_results(ret_df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark_etf_only"].dropna()
    common = strat.index.intersection(bench.index)
    nav_s = (1 + strat[common]).cumprod()
    nav_b = (1 + bench[common]).cumprod()
    excess_cum = nav_s / nav_b  # 相对净值

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax1, ax2 = axes

    ax1.plot(nav_s.index, nav_s.values, color="#2196F3", linewidth=1.8, label="混合策略")
    ax1.plot(nav_b.index, nav_b.values, color="#9E9E9E", linewidth=1.5,
             linestyle="--", label="纯ETF轮动基线")
    ax1.axvline(pd.Timestamp(OOS_START), color="red", linestyle="--", alpha=0.5, lw=1)
    ax1.set_title("ETF轮动 + 行业内选股 混合策略 vs 纯ETF基线（2016-2026）")
    ax1.set_ylabel("累积净值")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4)

    ax2.plot(excess_cum.index, excess_cum.values, color="#43A047", linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(pd.Timestamp(OOS_START), color="red", linestyle="--", alpha=0.5, lw=1)
    ax2.set_ylabel("超额净值（混合/ETF基线）")
    ax2.set_title("混合策略相对超额净值")
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path = out_dir / "etf_stock_hybrid_nav.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"净值图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────────

def main():
    out_dir = pathlib.Path(__file__).parent / "results" / "etf_stock_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载ETF价格
    print("加载ETF价格数据...")
    etf_close_full = load_close_matrix()
    etf_close = etf_close_full[etf_close_full.index >= START_DATE]
    valid_codes = [c for c in etf_close.columns
                   if etf_close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    etf_close = etf_close[valid_codes]
    print(f"  有效ETF：{len(valid_codes)} 只，区间：{etf_close.index[0].date()} ~ {etf_close.index[-1].date()}")

    # 2. 计算ETF动量得分和拥挤度
    print("\n计算ETF动量得分...")
    etf_scores = calc_etf_scores(etf_close)

    print("\n计算行业拥挤度...")
    crowding_pct = calc_crowding(etf_close)

    # 3. 加载个股价格（中证500成分股）
    print("\n加载中证500成分股价格...")
    if not HS500_MEMBERS_FILE.exists():
        raise FileNotFoundError(f"中证500成分股快照不存在：{HS500_MEMBERS_FILE}，请先运行 fetch_index_members.py")
    members_snap = pd.read_parquet(HS500_MEMBERS_FILE)
    all_stock_codes = members_snap["con_code"].unique().tolist()
    print(f"  中证500历史成分股：{len(all_stock_codes)} 只")

    stock_panel = load_close_panel(codes=all_stock_codes)
    print(f"  股票面板：{stock_panel.shape}  {stock_panel.index[0].date()} ~ {stock_panel.index[-1].date()}")

    # 4. 预加载财务数据（缓存）
    print("\n预加载财务数据...")
    for i, code in enumerate(all_stock_codes, 1):
        get_fina(code)
        if i % 200 == 0:
            print(f"  财务缓存：{i}/{len(all_stock_codes)}")
    print(f"  财务缓存完成，共 {len(_fina_cache)} 只")

    # 5. 加载行业映射
    print("\n加载申万行业映射...")
    industry_map = get_sw_industry_map()
    print(f"  行业映射：{len(industry_map)} 只")

    # 6. 运行混合策略回测
    print("\n开始混合策略回测...")
    ret_df = run_hybrid_backtest(
        etf_close=etf_close,
        etf_scores=etf_scores,
        crowding_pct=crowding_pct,
        stock_panel=stock_panel,
        hs500_members_file=HS500_MEMBERS_FILE,
        industry_map=industry_map,
    )

    if ret_df.empty:
        print("回测结果为空，请检查数据")
        return

    ret_df.to_csv(out_dir / "ret_detail.csv")
    print(f"\n回测完成，共 {len(ret_df)} 个月")

    # 7. 统计输出
    print("\n" + "=" * 65)
    print("混合策略 vs 纯ETF轮动基线（夏普1.005，年化17%，回撤-25.8%）")
    print("=" * 65)

    is_df  = ret_df[ret_df.index <= IS_END]
    oos_df = ret_df[ret_df.index >= OOS_START]

    print_stats(ret_df, "混合策略", "全样本（2016-2026）")
    print_stats(is_df,  "混合策略", f"IS（2016-{IS_END[:4]}）")
    print_stats(oos_df, "混合策略", f"OOS（{OOS_START[:4]}-2026）")
    print_annual_excess(ret_df)

    # 8. 槽位类型统计
    print("\n  槽位类型统计（行业选股 vs ETF直持 vs 降级）：")
    type_counts = {"stocks": 0, "etf": 0, "etf_fallback": 0}
    for slots in ret_df["slots"]:
        for s in slots:
            t = s.get("type", "etf")
            type_counts[t] = type_counts.get(t, 0) + 1
    total_slots = sum(type_counts.values())
    for t, cnt in type_counts.items():
        pct = cnt / total_slots * 100 if total_slots > 0 else 0
        print(f"    {t}: {cnt} ({pct:.1f}%)")

    # 9. 绘图
    plot_results(ret_df, out_dir)

    # 10. 验收判断
    print("\n" + "=" * 65)
    print("验收结论（vs纯ETF基线）：")
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark_etf_only"].dropna()
    common = strat.index.intersection(bench.index)
    excess_full = (strat[common] - bench[common]).mean() * 12 * 100

    is_strat = ret_df.loc[ret_df.index <= IS_END, "strategy"].dropna()
    is_bench = ret_df.loc[ret_df.index <= IS_END, "benchmark_etf_only"].dropna()
    ic = is_strat.index.intersection(is_bench.index)
    excess_is = (is_strat[ic] - is_bench[ic]).mean() * 12 * 100

    oos_strat = ret_df.loc[ret_df.index >= OOS_START, "strategy"].dropna()
    oos_bench = ret_df.loc[ret_df.index >= OOS_START, "benchmark_etf_only"].dropna()
    oc = oos_strat.index.intersection(oos_bench.index)
    excess_oos = (oos_strat[oc] - oos_bench[oc]).mean() * 12 * 100

    print(f"  全样本超额：{excess_full:+.1f}%  {'✓ 通过' if excess_full > 0 else '✗ 未通过（目标>0）'}")
    print(f"  IS超额：   {excess_is:+.1f}%  {'✓ 通过' if excess_is > 0 else '✗ 未通过（目标>0）'}")
    print(f"  OOS超额：  {excess_oos:+.1f}%  {'✓ 通过' if excess_oos > -2 else '✗ 未通过（目标>-2%）'}")

    print(f"\n输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
