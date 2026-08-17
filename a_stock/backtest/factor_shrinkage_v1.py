"""
因子权重收缩正则化 V1（指数增强新方向：介于固定ICIR加权和已证伪的滚动硬切换之间）

背景：lessons.md记录"自适应权重IS好/OOS差=过拟合信号"——此前滚动IC加权/regime
硬切换（完全跟随最近窗口ICIR）在IS上改善明显但OOS大幅衰退，根因是A股风格切换
剧烈不可预测。但完全固定ICIR（V2当前线上配置）无法响应任何结构性变化。

本版本用收缩估计量（James-Stein风格）代替"全固定"和"全跟随"两个极端：
  shrink_f(t) = rolling_icir_f(t)² / (rolling_icir_f(t)² + τ)
  weight_f(t) = shrink_f(t) * rolling_icir_f(t) + (1 - shrink_f(t)) * fixed_icir_f

rolling_icir高（滚动窗口内信号稳定）→ shrink→1，权重跟随滚动估计；
rolling_icir低或数据不足（噪音大/不稳定）→ shrink→0，权重收缩回固定基线。
τ越大，越保守（越接近固定ICIR加权）；τ越小，越激进（越接近失败的纯滚动方案）。

用τ敏感性扫描寻找是否存在使OOS优于固定基线的τ取值区间，而不是预设"收缩一定
有效"——如果所有τ取值都不优于固定基线，说明A股风格切换的不可预测性连"部分
跟随"都无法消化，应放弃此方向。

用法：
  cd a_stock/backtest
  python factor_shrinkage_v1.py --index hs500
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_ic_quality_v2 import get_industry_map
from factor_multi_backtest_v2 import (
    compute_reversal, compute_ep_sector, compute_ocf, compute_roe,
    compute_profit_stability, compute_composite_score,
    sharpe, max_drawdown, annual_return, get_fina,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_shrinkage_v1"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

IS_END    = "2024-01-31"
OOS_START = "2024-02-01"

MIN_STOCKS_CROSS = 50

COST_PER_TRADE   = 0.0030
STAMP_DUTY       = 0.001
RISK_FREE_ANNUAL = 0.02

ROLLING_WINDOW = 24   # 滚动ICIR窗口（月）
ROLLING_MIN_OBS = 12

TAU_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

FACTOR_NAMES = ["reversal", "ep_sector", "ocf", "roe", "profit_stability"]

INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
        "factor_icir": {
            "profit_stability": 0.322, "ep_sector": 0.321,
            "ocf": 0.219, "roe": 0.196, "reversal": 0.101,
        },
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
        "factor_icir": {
            "profit_stability": 0.322, "ep_sector": 0.321,
            "ocf": 0.219, "roe": 0.195, "reversal": 0.123,
        },
    },
}


# ── 截面工具（与V2一致） ──────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def get_monthly_ends(close_panel: pd.DataFrame) -> list[pd.Timestamp]:
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return [pd.Timestamp(d) for d in monthly_last]


def compute_raw_factor(fname: str, close_panel: pd.DataFrame, codes: list[str],
                       month_end: pd.Timestamp, close_row: pd.Series,
                       industry_map: dict) -> pd.Series:
    if fname == "reversal":
        return compute_reversal(close_panel, codes, month_end)
    elif fname == "ep_sector":
        return compute_ep_sector(codes, month_end, close_row, industry_map)
    elif fname == "ocf":
        return compute_ocf(codes, month_end)
    elif fname == "roe":
        return compute_roe(codes, month_end)
    elif fname == "profit_stability":
        return compute_profit_stability(codes, month_end)
    else:
        raise ValueError(fname)


# ── 逐因子月度IC序列（用于滚动ICIR估计） ────────────────────

def compute_factor_ic_series(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    industry_map: dict,
) -> pd.DataFrame:
    """
    每月截面：单因子标准化值 vs 下月收益率的Rank IC。
    返回 DataFrame，index=month_end，columns=因子名。
    """
    monthly_ends = get_monthly_ends(close_panel)
    records = []

    for i, month_end in enumerate(monthly_ends[:-1]):
        next_end = monthly_ends[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        close_row = close_panel[available].loc[month_end].dropna()
        codes = list(close_row.index)

        fwd_prices = close_panel[available].loc[next_end].dropna()
        price_common = close_row.index.intersection(fwd_prices.index)
        fwd_ret_all = fwd_prices[price_common] / close_row[price_common] - 1

        row = {"date": month_end}
        for fname in FACTOR_NAMES:
            raw = compute_raw_factor(fname, close_panel, codes, month_end, close_row, industry_map)
            if len(raw) < MIN_STOCKS_CROSS // 2:
                row[fname] = np.nan
                continue
            common = raw.index.intersection(fwd_ret_all.index)
            if len(common) < MIN_STOCKS_CROSS // 2:
                row[fname] = np.nan
                continue
            ic, _ = spearmanr(raw[common], fwd_ret_all[common])
            row[fname] = ic
        records.append(row)

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 滚动ICIR + 收缩权重 ──────────────────────────────────

def compute_rolling_icir(ic_df: pd.DataFrame, window: int = ROLLING_WINDOW,
                         min_obs: int = ROLLING_MIN_OBS) -> pd.DataFrame:
    """
    因果滚动ICIR：t月权重只用t月之前（不含t月）的IC历史，避免前视偏差。
    """
    icir = pd.DataFrame(index=ic_df.index, columns=ic_df.columns, dtype=float)
    for i in range(len(ic_df)):
        if i < min_obs:
            continue
        row_date = ic_df.index[i]
        window_ic = ic_df.iloc[max(0, i - window):i]
        for fname in ic_df.columns:
            clean = window_ic[fname].dropna()
            if len(clean) < min_obs or clean.std() < 1e-8:
                continue
            icir.loc[row_date, fname] = clean.mean() / clean.std()   # 用.loc避免链式赋值不生效
    return icir


def compute_shrinkage_weights(rolling_icir_row: pd.Series, fixed_icir: dict,
                              tau: float) -> dict[str, float]:
    """
    weight_f = shrink_f * rolling_icir_f + (1 - shrink_f) * fixed_icir_f
    shrink_f = rolling_icir_f² / (rolling_icir_f² + tau)
    负值裁剪为0后归一化（避免噪音驱动的符号反转破坏组合逻辑）。
    """
    weights = {}
    for fname, fixed_w in fixed_icir.items():
        r_icir = rolling_icir_row.get(fname, np.nan)
        if pd.isna(r_icir):
            weights[fname] = fixed_w   # 数据不足→完全用固定基线
            continue
        shrink = r_icir**2 / (r_icir**2 + tau)
        w = shrink * r_icir + (1 - shrink) * fixed_w
        weights[fname] = max(w, 0.0)

    total = sum(weights.values())
    if total < 1e-8:
        return fixed_icir   # 全部收缩到0，回退固定基线
    return {f: w / total for f, w in weights.items()}


# ── 组合得分（复用因子计算，权重方案可插拔） ─────────────────

def compute_composite_score_custom(
    close_panel: pd.DataFrame, codes: list[str], month_end: pd.Timestamp,
    weights: dict[str, float], industry_map: dict,
) -> pd.Series:
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    norm = {}
    for fname in FACTOR_NAMES:
        raw = compute_raw_factor(fname, close_panel, available, month_end, close_row, industry_map)
        if len(raw) < MIN_STOCKS_CROSS // 2:
            continue
        norm[fname] = standardize(winsorize(raw))

    if not norm:
        return pd.Series(dtype=float)

    common = None
    for fs in norm.values():
        common = set(fs.index) if common is None else common & set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return pd.Series(dtype=float)
    common = list(common)

    total_w = sum(weights.get(f, 0) for f in norm)
    if total_w < 1e-8:
        return pd.Series(dtype=float)

    score = pd.Series(0.0, index=common)
    for fname, fs in norm.items():
        score += fs[common] * (weights.get(fname, 0) / total_w)
    return score.dropna()


# ── 月度回测（对比：固定ICIR基线 vs 各τ收缩方案） ────────────

def run_backtest_shrinkage(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    fixed_icir: dict[str, float],
    industry_map: dict,
    rolling_icir: pd.DataFrame,
    tau: float | None,   # None = 固定基线
    top_n: int = 30,
) -> pd.DataFrame:
    monthly_ends = get_monthly_ends(close_panel)
    records = []

    for i, month_end in enumerate(monthly_ends[:-1]):
        next_end = monthly_ends[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        if tau is None:
            weights = fixed_icir
        else:
            if month_end not in rolling_icir.index:
                weights = fixed_icir
            else:
                weights = compute_shrinkage_weights(rolling_icir.loc[month_end], fixed_icir, tau)

        score = compute_composite_score_custom(close_panel, available, month_end, weights, industry_map)
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

        gross_ret = np.mean(ret_list)
        turnover_est = 0.5
        cost = turnover_est * (COST_PER_TRADE + STAMP_DUTY)

        bm_rets = []
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        benchmark_ret = np.mean(bm_rets) if bm_rets else np.nan

        records.append({
            "date": month_end,
            "strategy": gross_ret - cost,
            "benchmark": benchmark_ret,
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计 ──────────────────────────────────────────────

def compute_stats(ret_df: pd.DataFrame) -> dict:
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark"].dropna()
    common = strat.index.intersection(bench.index)
    strat, bench = strat[common], bench[common]
    excess = strat - bench
    nav = (1 + strat).cumprod()
    return {
        "样本月数": len(strat),
        "年化超额": excess.mean() * 12,
        "夏普": sharpe(strat),
        "最大回撤": max_drawdown(nav),
        "月胜率": (excess > 0).mean(),
    }


def print_period_stats(ret_df: pd.DataFrame, label: str) -> None:
    is_df  = ret_df[ret_df.index <= IS_END]
    oos_df = ret_df[ret_df.index >= OOS_START]
    for period_label, sub in [("全样本", ret_df), ("IS", is_df), ("OOS", oos_df)]:
        s = compute_stats(sub)
        print(f"    [{label} — {period_label}] 超额={s['年化超额']*100:+.2f}%/年  "
              f"夏普={s['夏普']:.3f}  MDD={s['最大回撤']*100:.1f}%  "
              f"月胜率={s['月胜率']*100:.1f}%  n={s['样本月数']}")


def plot_tau_sensitivity(results: dict, output_dir: pathlib.Path, title: str) -> None:
    """τ敏感性：横轴τ，纵轴IS/OOS年化超额"""
    taus = sorted([t for t in results if t != "fixed"])
    is_excess, oos_excess = [], []
    for t in taus:
        df = results[t]
        is_df  = df[df.index <= IS_END]
        oos_df = df[df.index >= OOS_START]
        is_excess.append(compute_stats(is_df)["年化超额"] * 100)
        oos_excess.append(compute_stats(oos_df)["年化超额"] * 100)

    fixed_is  = compute_stats(results["fixed"][results["fixed"].index <= IS_END])["年化超额"] * 100
    fixed_oos = compute_stats(results["fixed"][results["fixed"].index >= OOS_START])["年化超额"] * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
        lbl.set_color("white")

    ax.plot(taus, is_excess, marker="o", color="#60a5fa", label="收缩方案 IS超额")
    ax.plot(taus, oos_excess, marker="o", color="#f97316", label="收缩方案 OOS超额")
    ax.axhline(fixed_is, color="#60a5fa", linestyle="--", linewidth=1, label=f"固定ICIR基线 IS={fixed_is:.2f}%")
    ax.axhline(fixed_oos, color="#f97316", linestyle="--", linewidth=1, label=f"固定ICIR基线 OOS={fixed_oos:.2f}%")
    ax.set_xscale("log")
    ax.set_xlabel("τ（收缩强度，越小越激进）")
    ax.set_ylabel("年化超额（%）")
    ax.set_title(f"{title} — τ敏感性扫描")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "tau_sensitivity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"τ敏感性图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame, top_n: int,
                  industry_map: dict) -> None:
    cfg          = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]
    fixed_icir   = cfg["factor_icir"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  计算逐因子月度IC序列（用于滚动ICIR估计）...")
    ic_df = compute_factor_ic_series(close_panel, members_file, industry_map)
    if ic_df.empty:
        print("  IC序列为空，跳过")
        return
    ic_df.to_csv(out_dir / "factor_ic_series.csv")

    rolling_icir = compute_rolling_icir(ic_df)

    print(f"\n  回测固定ICIR基线（Top{top_n}）...")
    results = {"fixed": run_backtest_shrinkage(close_panel, members_file, fixed_icir,
                                               industry_map, rolling_icir, None, top_n)}

    for tau in TAU_GRID:
        print(f"  回测收缩方案 τ={tau}（Top{top_n}）...")
        results[tau] = run_backtest_shrinkage(close_panel, members_file, fixed_icir,
                                              industry_map, rolling_icir, tau, top_n)

    print(f"\n{'='*70}")
    print(f"指数：{index_name}  Top{top_n}  因子权重收缩正则化 τ敏感性")
    print(f"{'='*70}")

    print_period_stats(results["fixed"], "固定ICIR基线")
    for tau in TAU_GRID:
        if not results[tau].empty:
            print_period_stats(results[tau], f"收缩 τ={tau}")

    valid_results = {k: v for k, v in results.items() if not v.empty}
    if len(valid_results) > 1:
        plot_tau_sensitivity(valid_results, out_dir, f"{index_name} Top{top_n}")

    for tau, df in results.items():
        if not df.empty:
            df.to_csv(out_dir / f"ret_{tau}.csv")


def main():
    parser = argparse.ArgumentParser(description="因子权重收缩正则化 V1")
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
