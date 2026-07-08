"""
多因子选股回测 V2（盈利质量因子组合）
- 因子：盈利增速稳定性 + 行业内EP + OCF/NI + ROE_TTM + 反转
- 去掉上版的低波动因子（A股牛市系统性反向）
- 行业内EP：在申万行业内截面排名，消除行业配置暴露

用法：
  cd a_stock/backtest
  python factor_multi_backtest_v2.py --index hs500
  python factor_multi_backtest_v2.py --index hs500 --top 20
"""

import sys
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR
from fetch_financials import load_financials

# 从 v2 IC 脚本复用资产负债表加载（不需要 accruals 了，但 industry map 需要）
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_ic_quality_v2 import (
    get_industry_map,
    load_balancesheet,
    BS_DIR,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_multi_backtest_v2"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_CROSS   = 50
MIN_STOCKS_SECTOR  = 5    # 行业内 EP 最少股票数
MIN_HISTORY_QTRS   = 4    # 盈利增速稳定性最少季度数

IS_END    = "2024-01-31"
OOS_START = "2024-02-01"

COST_PER_TRADE   = 0.0030   # 佣金+滑点（双边）
STAMP_DUTY       = 0.001    # 印花税（卖出）
RISK_FREE_ANNUAL = 0.02

REVERSAL_WINDOW = 63   # 中证500 最优

# 新因子 ICIR 权重（来自 factor_ic_quality_v2 验证结果）
INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
        "factor_icir": {
            "profit_stability": 0.322,  # 暂用 hs500 值，待验证
            "ep_sector":        0.321,
            "ocf":              0.219,
            "roe":              0.196,
            "reversal":         0.101,
        },
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
        "factor_icir": {
            "profit_stability": 0.322,
            "ep_sector":        0.321,
            "ocf":              0.219,
            "roe":              0.195,
            "reversal":         0.123,
        },
    },
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


def get_fina_history(ts_code: str, as_of: pd.Timestamp,
                     field: str, n: int = 8) -> pd.Series:
    df = get_fina(ts_code)
    if df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    return valid[field].iloc[-n:].reset_index(drop=True)


# ── 因子计算 ──────────────────────────────────────────────

def compute_reversal(close_panel: pd.DataFrame, codes: list[str],
                     month_end: pd.Timestamp, window: int = REVERSAL_WINDOW) -> pd.Series:
    hist = close_panel[codes].loc[:month_end]
    if len(hist) < window + 1:
        return pd.Series(dtype=float)
    ret = hist.iloc[-1] / hist.iloc[-(window + 1)] - 1
    return -ret.dropna()   # 取负：跌多 → 高得分


def compute_ep_sector(codes: list[str], month_end: pd.Timestamp,
                      close_row: pd.Series, industry_map: dict) -> pd.Series:
    """行业内 EP 分位排名（0-1），消除行业配置暴露"""
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

    ranks = {}
    fallback_rank = ep_series.rank(pct=True)  # 兜底：行业太小时用全截面排名
    for code in ep_series.index:
        sector = sector_series[code]
        if sector == "未知":
            ranks[code] = fallback_rank[code]
            continue
        group = ep_series[sector_series == sector]
        if len(group) < MIN_STOCKS_SECTOR:
            ranks[code] = fallback_rank[code]
        else:
            ranks[code] = group.rank(pct=True)[code]

    return pd.Series(ranks)


def compute_ocf(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """OCF / 净利润（现金流质量）"""
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "ocf_to_profit")
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


def compute_roe(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """ROE_TTM"""
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "roe_dt")
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


def compute_profit_stability(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """盈利增速稳定性：过去 8 季度 netprofit_yoy 标准差取负（稳定=高得分）"""
    values = {}
    for code in codes:
        hist = get_fina_history(code, month_end, "netprofit_yoy", n=8)
        if len(hist) < MIN_HISTORY_QTRS:
            continue
        std = hist.std()
        if pd.notna(std) and std >= 0:
            values[code] = -std    # 取负：越稳定得分越高
    return pd.Series(values)


# ── 截面标准化工具 ─────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


# ── 综合得分合成 ───────────────────────────────────────────

def compute_composite_score(
    close_panel: pd.DataFrame,
    codes: list[str],
    month_end: pd.Timestamp,
    icir_weights: dict[str, float],
    industry_map: dict,
    weight_scheme: str = "icir",
) -> pd.Series:
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    raw_factors = {
        "reversal":         compute_reversal(close_panel, available, month_end),
        "ep_sector":        compute_ep_sector(available, month_end, close_row, industry_map),
        "ocf":              compute_ocf(available, month_end),
        "roe":              compute_roe(available, month_end),
        "profit_stability": compute_profit_stability(available, month_end),
    }

    # 截面标准化
    norm = {}
    for fname, fs in raw_factors.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm[fname] = fs

    if not norm:
        return pd.Series(dtype=float)

    # 求公共股票集
    common = None
    for fs in norm.values():
        if common is None:
            common = set(fs.index)
        else:
            common &= set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return pd.Series(dtype=float)
    common = list(common)

    # 加权
    if weight_scheme == "icir":
        total_w = sum(icir_weights.get(f, 0) for f in norm)
        if total_w < 1e-8:
            return pd.Series(dtype=float)
        score = pd.Series(0.0, index=common)
        for fname, fs in norm.items():
            w = icir_weights.get(fname, 0) / total_w
            score += fs[common] * w
    else:
        score = pd.Series(0.0, index=common)
        for fs in norm.values():
            score += fs[common] / len(norm)

    return score.dropna()


# ── 月度回测 ──────────────────────────────────────────────

def run_backtest(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    icir_weights: dict[str, float],
    industry_map: dict,
    top_n: int = 30,
    weight_scheme: str = "icir",
) -> pd.DataFrame:
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_end = monthly_last[i + 1]
        month_end = pd.Timestamp(month_end)
        next_end  = pd.Timestamp(next_end)

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        score = compute_composite_score(
            close_panel, available, month_end,
            icir_weights, industry_map, weight_scheme
        )
        if len(score) < top_n:
            continue

        selected = score.nlargest(top_n).index.tolist()

        ret_list = []
        for code in selected:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret_list.append(p1 / p0 - 1)

        if not ret_list:
            continue

        gross_ret    = np.mean(ret_list)
        turnover_est = 0.5
        cost         = turnover_est * (COST_PER_TRADE + STAMP_DUTY)
        strategy_ret = gross_ret - cost

        bm_rets = []
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        benchmark_ret = np.mean(bm_rets) if bm_rets else np.nan

        records.append({
            "date":      month_end,
            "strategy":  strategy_ret,
            "benchmark": benchmark_ret,
            "gross_ret": gross_ret,
            "n_stocks":  len(ret_list),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计 ──────────────────────────────────────────────

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
    bench = ret_df["benchmark"].dropna()
    common = strat.index.intersection(bench.index)
    strat, bench = strat[common], bench[common]
    excess = strat - bench

    nav_s = (1 + strat).cumprod()
    nav_b = (1 + bench).cumprod()

    print(f"\n  [{label} — {period}]")
    print(f"    期间: {period}")
    print(f"    样本月数: {len(strat)}")
    print(f"    策略年化: {annual_return(nav_s)*100:.1f}%")
    print(f"    基准年化: {annual_return(nav_b)*100:.1f}%")
    print(f"    超额年化（gross）: {excess.mean()*12*100:.1f}%")
    print(f"    策略夏普: {sharpe(strat):.3f}")
    print(f"    基准夏普: {sharpe(bench):.3f}")
    print(f"    策略最大回撤: {max_drawdown(nav_s)*100:.1f}%")
    print(f"    基准最大回撤: {max_drawdown(nav_b)*100:.1f}%")
    print(f"    月胜率vs基准: {(excess > 0).mean()*100:.1f}%")


def print_annual_excess(ret_df: pd.DataFrame, label: str) -> None:
    """年度超额分拆（gross），用于诊断牛市/熊市行为"""
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark"].dropna()
    excess = (strat - bench).dropna()

    print(f"\n  {label} 年度超额 gross（月均）:")
    for y in sorted(excess.index.year.unique()):
        yr = excess[excess.index.year == y]
        print(f"    {y}: {yr.mean()*100:+.2f}%/月  (n={len(yr)})")


def plot_nav(results: dict, output_dir: pathlib.Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    colors = ["#60a5fa", "#f97316", "#34d399", "#facc15"]

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    ax_nav, ax_excess = axes
    bm_plotted = False

    for (label, ret_df), color in zip(results.items(), colors):
        strat = ret_df["strategy"].dropna()
        bench = ret_df["benchmark"].dropna()
        nav_s = (1 + strat).cumprod()
        nav_b = (1 + bench).cumprod()
        excess_cum = (1 + strat - bench).cumprod()

        ax_nav.plot(nav_s.index, nav_s.values, color=color, linewidth=1.5, label=label)
        if not bm_plotted:
            ax_nav.plot(nav_b.index, nav_b.values, color="gray",
                        linewidth=1.2, linestyle="--", label="基准（等权）")
            bm_plotted = True
        ax_excess.plot(excess_cum.index, excess_cum.values,
                       color=color, linewidth=1.5, label=label)

    ax_nav.set_title(f"{title} — 净值曲线")
    ax_nav.set_ylabel("净值")
    ax_nav.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_excess.set_title(f"{title} — 超额累积净值")
    ax_excess.set_ylabel("超额累积净值")
    ax_excess.axhline(1, color="white", linewidth=0.8, linestyle="--")
    ax_excess.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_excess.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "nav.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"净值图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame,
                  top_n: int, industry_map: dict) -> None:
    cfg          = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]
    icir_weights = cfg["factor_icir"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  回测 ICIR 加权（Top{top_n}）...")
    ret_icir = run_backtest(close_panel, members_file, icir_weights,
                            industry_map, top_n, "icir")

    print(f"  回测等权合成（Top{top_n}）...")
    ret_equal = run_backtest(close_panel, members_file, icir_weights,
                             industry_map, top_n, "equal")

    if ret_icir.empty:
        print("  无有效回测结果")
        return

    ret_icir.to_csv(out_dir / "ret_icir.csv")
    ret_equal.to_csv(out_dir / "ret_equal.csv")

    print(f"\n{'='*60}")
    print(f"指数：{index_name}  Top{top_n}  新因子组合（无低波动）")
    print(f"{'='*60}")

    for label, ret_df in [("ICIR加权", ret_icir), ("等权合成", ret_equal)]:
        is_df  = ret_df[ret_df.index <= IS_END]
        oos_df = ret_df[ret_df.index >= OOS_START]
        print_stats(ret_df,  label, "全样本")
        print_stats(is_df,   label, "IS（2016-2024）")
        print_stats(oos_df,  label, "OOS（2024-2026）")

    print_annual_excess(ret_icir, "ICIR加权")

    plot_nav({"ICIR加权": ret_icir, "等权合成": ret_equal},
             out_dir, f"{index_name} Top{top_n} V2")

    print(f"\n  TopN 敏感性（ICIR加权）...")
    for n in [10, 20, 50]:
        r = run_backtest(close_panel, members_file, icir_weights,
                         industry_map, n, "icir")
        if r.empty:
            continue
        s   = sharpe(r["strategy"].dropna())
        mdd = max_drawdown((1 + r["strategy"].dropna()).cumprod())
        ann = annual_return((1 + r["strategy"].dropna()).cumprod())
        print(f"    Top{n}: 年化={ann*100:.1f}%  夏普={s:.3f}  最大回撤={mdd*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="多因子选股回测 V2（盈利质量组合）")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="hs500")
    parser.add_argument("--top",   type=int, default=30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    all_codes: set[str] = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    print("预加载财务数据...")
    for i, code in enumerate(all_codes, 1):
        get_fina(code)
        if i % 200 == 0:
            print(f"  财务缓存：{i}/{len(all_codes)}")
    print()

    print("加载行业映射...")
    industry_map = get_industry_map()
    print(f"  行业映射：{len(industry_map)} 只\n")

    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        run_one_index(key, close_panel, args.top, industry_map)

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
