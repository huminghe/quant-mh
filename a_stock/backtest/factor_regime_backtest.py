"""
市场状态（Factor Regime）条件切换回测

核心思路：用每个因子近期滚动 IC 判断当前因子是否处于有效状态，
动态调整合成权重，而非固定 ICIR 权重。

两种方案对比：
  方案A：滚动 IC 加权（近12月 IC 均值作为当期权重，IC 反向时降权至0）
  方案B：硬切换（若某因子近12月 IC < 阈值，该因子退出合成，只用有效因子）

基线：V2 线性固定 ICIR 加权

用法：
  cd a_stock/backtest
  python factor_regime_backtest.py
"""

import sys
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_ic_quality_v2 import get_industry_map
from factor_multi_backtest_v2 import (
    get_fina, get_fina_pit, get_fina_history,
    compute_reversal, compute_ep_sector, compute_ocf,
    compute_roe, compute_profit_stability,
    winsorize, standardize,
    MIN_STOCKS_CROSS, INDEX_CONFIG,
    COST_PER_TRADE, STAMP_DUTY, RISK_FREE_ANNUAL,
    run_backtest as run_linear_backtest,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR   = pathlib.Path(__file__).parent / "results" / "factor_regime_backtest"
MEMBERS_FILE = INDEX_CONFIG["hs500"]["members_file"]
ICIR_WEIGHTS = INDEX_CONFIG["hs500"]["factor_icir"]   # 固定权重基线

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"
IS_END     = "2024-01-31"
OOS_START  = "2024-02-01"

TOP_N = 30

# 滚动 IC 窗口（月）：用过去多少期 IC 估计当前因子有效性
IC_WINDOW = 12

# 方案B 阈值：近12月均值 IC < 此值时该因子退出
IC_CUTOFF = 0.0   # IC 负向时退出（保守）

FACTOR_NAMES = ["profit_stability", "ep_sector", "ocf", "roe", "reversal"]


# ── 截面因子计算（复用） ──────────────────────────────────

def get_factor_scores(close_panel, codes, month_end, industry_map):
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    raw = {
        "reversal":         compute_reversal(close_panel, available, month_end),
        "ep_sector":        compute_ep_sector(available, month_end, close_row, industry_map),
        "ocf":              compute_ocf(available, month_end),
        "roe":              compute_roe(available, month_end),
        "profit_stability": compute_profit_stability(available, month_end),
    }

    norm = {}
    for fname, fs in raw.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm[fname] = fs

    common = None
    for fs in norm.values():
        common = set(fs.index) if common is None else common & set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return None, None

    return norm, list(common)


def compute_cross_ic(factor_scores: pd.Series, fwd_ret: pd.Series) -> float:
    common = factor_scores.index.intersection(fwd_ret.index)
    if len(common) < 20:
        return np.nan
    ic, _ = spearmanr(factor_scores[common], fwd_ret[common])
    return ic


# ── 主回测（支持两种 regime 方案） ───────────────────────

def run_regime_backtest(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    industry_map: dict,
    scheme: str = "rolling_ic",   # "rolling_ic" 或 "hard_cutoff"
    ic_window: int = IC_WINDOW,
    ic_cutoff: float = IC_CUTOFF,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    close_sub    = close_panel.loc[START_DATE:END_DATE]
    nat_ends     = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    # 历史 IC 队列：{factor_name: deque of recent IC values}
    from collections import deque
    ic_history = {f: deque(maxlen=ic_window) for f in FACTOR_NAMES}

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_end  = monthly_last[i + 1]
        month_end = pd.Timestamp(month_end)
        next_end  = pd.Timestamp(next_end)

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        norm, common = get_factor_scores(close_panel, available, month_end, industry_map)
        if norm is None:
            continue

        # 未来收益（用于计算本月 IC，供下期使用）
        fwd_ret_dict = {}
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret_dict[code] = p1 / p0 - 1
        fwd_ret = pd.Series(fwd_ret_dict)

        # ── 当期权重计算 ──────────────────────────────────
        if scheme == "rolling_ic":
            # 用近 ic_window 期滚动 IC 均值作为权重；IC 负时截断为 0
            weights = {}
            for fname in FACTOR_NAMES:
                if fname not in norm:
                    weights[fname] = 0.0
                    continue
                hist = list(ic_history[fname])
                if len(hist) < 6:
                    # 历史不足时，用固定 ICIR 权重兜底
                    weights[fname] = ICIR_WEIGHTS.get(fname, 0.1)
                else:
                    roll_ic = np.nanmean(hist)
                    weights[fname] = max(roll_ic, 0.0)   # 负 IC 截断为 0

        elif scheme == "hard_cutoff":
            # 近 ic_window 期均值 < ic_cutoff 时该因子退出合成
            weights = {}
            for fname in FACTOR_NAMES:
                if fname not in norm:
                    weights[fname] = 0.0
                    continue
                hist = list(ic_history[fname])
                if len(hist) < 6:
                    weights[fname] = ICIR_WEIGHTS.get(fname, 0.1)
                else:
                    roll_ic = np.nanmean(hist)
                    if roll_ic < ic_cutoff:
                        weights[fname] = 0.0          # 退出
                    else:
                        weights[fname] = ICIR_WEIGHTS.get(fname, 0.1)  # 保持原权重

        total_w = sum(weights.values())
        if total_w < 1e-8:
            # 所有因子都退出时，回退到等权
            weights = {f: 1.0 / len(FACTOR_NAMES) for f in FACTOR_NAMES}
            total_w = 1.0

        # ── 合成得分 ──────────────────────────────────────
        score = pd.Series(0.0, index=common)
        active_factors = []
        for fname, fs in norm.items():
            w = weights.get(fname, 0.0) / total_w
            if w > 1e-8:
                score += fs[common] * w
                active_factors.append(fname)

        if len(score) < top_n:
            continue

        selected = score.nlargest(top_n).index.tolist()

        # ── 收益计算 ──────────────────────────────────────
        ret_list = [fwd_ret[c] for c in selected if c in fwd_ret and pd.notna(fwd_ret[c])]
        if not ret_list:
            continue

        gross_ret    = np.mean(ret_list)
        cost         = 0.5 * (COST_PER_TRADE + STAMP_DUTY)
        strategy_ret = gross_ret - cost

        bm_codes = [c for c in available if c in fwd_ret]
        bm_ret   = fwd_ret[bm_codes].mean() if bm_codes else np.nan

        records.append({
            "date":           month_end,
            "strategy":       strategy_ret,
            "benchmark":      bm_ret,
            "gross_ret":      gross_ret,
            "n_active_factors": len(active_factors),
            **{f"w_{fn}": weights.get(fn, 0) / total_w for fn in FACTOR_NAMES},
        })

        # ── 更新 IC 历史（当期因子得分 vs 实际未来收益）─────
        for fname, fs in norm.items():
            ic = compute_cross_ic(fs, fwd_ret)
            if pd.notna(ic):
                ic_history[fname].append(ic)

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计 ──────────────────────────────────────────────

def sharpe(ret: pd.Series, freq: int = 12) -> float:
    if len(ret) < 2: return np.nan
    ann = ret.mean() * freq
    std = ret.std() * np.sqrt(freq)
    return np.nan if std < 1e-8 else (ann - RISK_FREE_ANNUAL) / std

def max_drawdown(nav: pd.Series) -> float:
    return ((nav - nav.cummax()) / nav.cummax()).min()

def annual_return(nav: pd.Series, freq: int = 12) -> float:
    n = len(nav)
    if n < 2: return np.nan
    return (1 + nav.iloc[-1] / nav.iloc[0] - 1) ** (1 / (n / freq)) - 1

def print_stats(ret_df: pd.DataFrame, label: str, start=None, end=None) -> None:
    df = ret_df.copy()
    if start: df = df[df.index >= start]
    if end:   df = df[df.index <= end]
    strat  = df["strategy"].dropna()
    bench  = df["benchmark"].dropna()
    common = strat.index.intersection(bench.index)
    strat, bench = strat[common], bench[common]
    excess = strat - bench
    nav_s = (1 + strat).cumprod()
    print(f"\n  [{label}]  n={len(strat)}")
    print(f"    策略年化：{annual_return(nav_s)*100:.1f}%  超额年化：{excess.mean()*12*100:+.1f}%")
    print(f"    策略夏普：{sharpe(strat):.3f}  月胜率：{(excess>0).mean()*100:.1f}%")
    print(f"    最大回撤：{max_drawdown(nav_s)*100:.1f}%")

def print_annual(ret_df: pd.DataFrame, label: str) -> None:
    strat  = ret_df["strategy"].dropna()
    bench  = ret_df["benchmark"].dropna()
    excess = (strat - bench).dropna()
    print(f"\n  {label} 年度超额（net，月均）:")
    for y in sorted(excess.index.year.unique()):
        yr = excess[excess.index.year == y]
        print(f"    {y}: {yr.mean()*100:+.2f}%/月  (n={len(yr)})")


def plot_all(results: dict, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    colors = ["#60a5fa", "#f97316", "#34d399", "#facc15"]
    for ax in axes:
        ax.set_facecolor("#16213e"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    ax_nav, ax_exc = axes
    bm_done = False
    for (label, ret_df), color in zip(results.items(), colors):
        strat  = ret_df["strategy"].dropna()
        bench  = ret_df["benchmark"].dropna()
        nav_s  = (1 + strat).cumprod()
        nav_b  = (1 + bench).cumprod()
        excess = (1 + strat - bench).cumprod()
        ax_nav.plot(nav_s.index, nav_s.values, color=color, lw=1.5, label=label)
        if not bm_done:
            ax_nav.plot(nav_b.index, nav_b.values, color="gray", lw=1.2, ls="--", label="基准")
            bm_done = True
        ax_exc.plot(excess.index, excess.values, color=color, lw=1.5, label=label)

    ax_nav.set_title("净值曲线"); ax_nav.set_ylabel("净值")
    ax_nav.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_exc.set_title("超额累积净值"); ax_exc.set_ylabel("超额净值")
    ax_exc.axhline(1, color="white", lw=0.8, ls="--")
    ax_exc.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_exc.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out = output_dir / "regime_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  图表已保存：{out}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    members_df = pd.read_parquet(MEMBERS_FILE)
    all_codes  = members_df["con_code"].unique().tolist()

    print("加载收盘价面板...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"面板：{close_panel.shape}")

    print("预加载财务数据...")
    for i, code in enumerate(all_codes, 1):
        get_fina(code)
        if i % 200 == 0:
            print(f"  {i}/{len(all_codes)}")

    print("加载行业映射...")
    industry_map = get_industry_map()
    print(f"  {len(industry_map)} 只\n")

    results = {}

    print("=" * 60)
    print("  方案A：滚动 IC 加权（12月窗口，负IC截断为0）")
    print("=" * 60)
    ret_a = run_regime_backtest(close_panel, MEMBERS_FILE, industry_map,
                                scheme="rolling_ic", ic_window=12)
    results["滚动IC加权"] = ret_a
    print_stats(ret_a, "全样本")
    print_stats(ret_a, "IS（2016-2024）", end=IS_END)
    print_stats(ret_a, "OOS（2024-2026）", start=OOS_START)
    print_annual(ret_a, "方案A")

    print("\n" + "=" * 60)
    print("  方案B：硬切换（近12月均值IC<0时退出，保持原权重比例）")
    print("=" * 60)
    ret_b = run_regime_backtest(close_panel, MEMBERS_FILE, industry_map,
                                scheme="hard_cutoff", ic_window=12, ic_cutoff=0.0)
    results["硬切换(IC<0退出)"] = ret_b
    print_stats(ret_b, "全样本")
    print_stats(ret_b, "IS（2016-2024）", end=IS_END)
    print_stats(ret_b, "OOS（2024-2026）", start=OOS_START)
    print_annual(ret_b, "方案B")

    # 线性基线（对比）
    print("\n" + "=" * 60)
    print("  线性 ICIR 基线（V2）")
    print("=" * 60)
    ret_linear = run_linear_backtest(close_panel, MEMBERS_FILE, ICIR_WEIGHTS,
                                     industry_map, TOP_N, "icir")
    results["线性ICIR基线"] = ret_linear
    print_stats(ret_linear, "全样本")
    print_stats(ret_linear, "IS（2016-2024）", end=IS_END)
    print_stats(ret_linear, "OOS（2024-2026）", start=OOS_START)
    print_annual(ret_linear, "线性ICIR")

    # 对比汇总
    print("\n" + "=" * 60)
    print("  汇总对比（IS 超额年化）")
    print("=" * 60)
    for label, ret_df in results.items():
        is_df  = ret_df[ret_df.index <= IS_END]
        strat  = is_df["strategy"].dropna()
        bench  = is_df["benchmark"].dropna()
        excess = (strat - bench.reindex(strat.index)).dropna()
        print(f"  {label:<20}  IS超额：{excess.mean()*12*100:+.1f}%/年  月胜率：{(excess>0).mean()*100:.1f}%")

    plot_all(results, OUTPUT_DIR)
    print(f"\n  输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
