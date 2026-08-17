"""
风险模型约束组合优化 V1（指数增强新方向：跳出固定权重TopN框架）

背景：此前5轮40+种方向全部在"因子合成分数→选TopN等权"框架内穷尽，从未引入
机构标准的风险模型约束组合优化。本版本用V2因子组合分数作为alpha，在
size/beta/行业暴露中性约束下用线性规划求解最优权重，替代简单TopN等权。

风险模型（简化版，非完整协方差矩阵）：
- size暴露：log(total_mv) 截面标准化，中性化到全universe均值附近
- beta暴露：63日对universe等权收益回归斜率（复用factor_ic_lowvol.py的leave-one-out回归），
  中性化到全universe均值附近（≈1）
- 行业暴露：申万行业dummy，每个行业权重中性化到全universe等权占比附近

优化问题是纯线性规划（LP）：目标=最大化持仓alpha分数（线性），约束全部线性
（权重求和=1、单票上限、暴露区间），用scipy.optimize.linprog(method="highs")
直接求全局最优解，不引入cvxpy新依赖。

局限：LP没有协方差风险项（不是真正的均值-方差优化），解会偏向"顶格/清零"的
角解，这是纯alpha-max+暴露约束的已知特征。若v1验证有效，再考虑加风险厌恶项
升级为QP（仍可用scipy.optimize.minimize做小规模SLSQP，无需cvxpy）。

用法：
  cd a_stock/backtest
  python factor_portfolio_opt_v1.py --index hs500
"""

import sys
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog
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
    compute_composite_score, sharpe, max_drawdown, annual_return,
    get_fina, winsorize, standardize,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_portfolio_opt_v1"
VALUATION_FILE = pathlib.Path(__file__).parent.parent / "data" / "valuation_monthly.parquet"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

IS_END    = "2024-01-31"
OOS_START = "2024-02-01"

MIN_STOCKS_CROSS = 50
MIN_STOCKS_SECTOR_OPT = 5   # 行业约束最少股票数，太小的行业不单独约束

COST_PER_TRADE   = 0.0030
STAMP_DUTY       = 0.001
RISK_FREE_ANNUAL = 0.02

W_MAX       = 0.05   # 单票权重上限
SIZE_TOL    = 0.15   # size暴露容差（标准化后的标准差单位）
BETA_TOL    = 0.15   # beta暴露容差
SECTOR_TOL  = 0.03   # 单行业权重容差（相对universe等权占比，绝对值）

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


# ── 市值数据（size暴露） ──────────────────────────────────

_valuation_cache: pd.DataFrame | None = None

def load_valuation() -> pd.DataFrame:
    global _valuation_cache
    if _valuation_cache is None:
        df = pd.read_parquet(VALUATION_FILE)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        _valuation_cache = df
    return _valuation_cache


def get_size_exposure(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """size暴露 = log(total_mv)，取月度调仓日快照（total_mv单位：万元）"""
    val = load_valuation()
    snap = val[val["trade_date"] == month_end]
    if snap.empty:
        # 找不到精确匹配日，取最近的前一个快照
        prior = val[val["trade_date"] <= month_end]
        if prior.empty:
            return pd.Series(dtype=float)
        last_date = prior["trade_date"].max()
        snap = val[val["trade_date"] == last_date]
    snap = snap.set_index("ts_code")
    sub = snap.loc[snap.index.intersection(codes), "total_mv"]
    sub = sub[sub > 0]
    return np.log(sub)


# ── beta暴露（复用factor_ic_lowvol.py的leave-one-out回归逻辑） ──

def get_beta_exposure(ret_panel: pd.DataFrame, codes: list[str],
                      month_end: pd.Timestamp, window: int = 63,
                      min_obs: int = 30) -> pd.Series:
    """
    beta暴露：过去window日日收益率对universe等权市场收益回归斜率。
    market代理与ivol计算一致：leave-one-out等权（排除自身避免虚高相关）。
    """
    hist = ret_panel[codes].loc[:month_end].iloc[-(window + 1):-1]
    if len(hist) < min_obs:
        return pd.Series(dtype=float)

    n_valid_per_day = hist.notna().sum(axis=1)
    col_sum = hist.sum(axis=1)

    betas = {}
    for code in hist.columns:
        stock_ret = hist[code].dropna()
        if len(stock_ret) < min_obs:
            continue
        n_valid = n_valid_per_day.loc[stock_ret.index]
        loo_market = (col_sum.loc[stock_ret.index] - stock_ret) / (n_valid - 1).clip(lower=1)
        X = np.column_stack([np.ones(len(loo_market)), loo_market.values])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, stock_ret.values, rcond=None)
            betas[code] = coef[1]
        except Exception:
            continue
    return pd.Series(betas)


# ── 组合优化器（LP：alpha-max + 暴露中性约束） ──────────────

def optimize_portfolio(
    alpha: pd.Series,          # index=ts_code，因子合成分数（越高越好）
    size_exp: pd.Series,       # index=ts_code，log(total_mv)
    beta_exp: pd.Series,       # index=ts_code
    sector_map: dict,          # ts_code -> 行业名
    w_max: float = W_MAX,
    size_tol: float = SIZE_TOL,
    beta_tol: float = BETA_TOL,
    sector_tol: float = SECTOR_TOL,
) -> pd.Series:
    """
    求解：max alpha^T w
    s.t.  sum(w) = 1, 0 <= w <= w_max
          |size_exp^T w - size_exp_universe_mean| <= size_tol * size_exp_std
          |beta_exp^T w - beta_exp_universe_mean| <= beta_tol * beta_exp_std
          对每个行业：|sector_weight - universe等权占比| <= sector_tol

    universe = alpha.index ∩ size_exp.index ∩ beta_exp.index 的公共股票集，
    暴露的中性基准取该universe的等权均值（而非alpha分数的目标持仓均值），
    这样约束的是"相对被动持有该universe"的风格暴露，不是相对TopN子集。
    """
    common = alpha.index.intersection(size_exp.index).intersection(beta_exp.index)
    if len(common) < MIN_STOCKS_CROSS:
        return pd.Series(dtype=float)
    common = list(common)
    n = len(common)

    a = alpha.loc[common].values
    sz = size_exp.loc[common].values
    bt = beta_exp.loc[common].values

    sz_std = sz.std()
    bt_std = bt.std()
    if sz_std < 1e-8 or bt_std < 1e-8:
        return pd.Series(dtype=float)

    sz_mean_uw = sz.mean()   # universe等权均值（中性化基准）
    bt_mean_uw = bt.mean()

    sectors = pd.Series({c: sector_map.get(c, "未知") for c in common})
    sector_counts = sectors.value_counts()
    valid_sectors = sector_counts[sector_counts >= MIN_STOCKS_SECTOR_OPT].index.tolist()

    # linprog 求最小值，取负alpha做最大化
    c = -a

    A_ub, b_ub = [], []

    # size暴露上下界
    A_ub.append(sz); b_ub.append(sz_mean_uw + size_tol * sz_std)
    A_ub.append(-sz); b_ub.append(-(sz_mean_uw - size_tol * sz_std))

    # beta暴露上下界
    A_ub.append(bt); b_ub.append(bt_mean_uw + beta_tol * bt_std)
    A_ub.append(-bt); b_ub.append(-(bt_mean_uw - beta_tol * bt_std))

    # 行业暴露上下界
    for sec in valid_sectors:
        mask = (sectors.values == sec).astype(float)
        uw_share = sector_counts[sec] / n   # universe等权占比
        A_ub.append(mask); b_ub.append(uw_share + sector_tol)
        A_ub.append(-mask); b_ub.append(-(uw_share - sector_tol))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    bounds = [(0, w_max)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")

    if not res.success:
        return pd.Series(dtype=float)

    w = pd.Series(res.x, index=common)
    return w[w > 1e-6]


# ── 月度回测 ──────────────────────────────────────────────

def get_monthly_ends(close_panel: pd.DataFrame) -> list[pd.Timestamp]:
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return [pd.Timestamp(d) for d in monthly_last]


def run_backtest_compare(
    close_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    members_file: pathlib.Path,
    icir_weights: dict[str, float],
    industry_map: dict,
    top_n: int = 30,
) -> pd.DataFrame:
    """同一月度截面下并行对比：TopN等权 vs 优化组合（相同alpha分数）"""
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

        score = compute_composite_score(
            close_panel, available, month_end,
            icir_weights, industry_map, "icir"
        )
        if len(score) < top_n:
            continue

        # --- TopN等权 ---
        selected = score.nlargest(top_n).index.tolist()
        topn_ret = []
        for code in selected:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                topn_ret.append(p1 / p0 - 1)
        if not topn_ret:
            continue
        topn_gross = np.mean(topn_ret)

        # --- 优化组合 ---
        size_exp = get_size_exposure(score.index.tolist(), month_end)
        beta_exp = get_beta_exposure(ret_panel, score.index.tolist(), month_end)
        w = optimize_portfolio(score, size_exp, beta_exp, industry_map)

        if w.empty:
            opt_gross = np.nan
        else:
            opt_ret = []
            opt_w = []
            for code, wt in w.items():
                p0 = close_panel[code].get(month_end)
                p1 = close_panel[code].get(next_end)
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    opt_ret.append(p1 / p0 - 1)
                    opt_w.append(wt)
            if opt_ret:
                opt_w = np.array(opt_w) / np.sum(opt_w)
                opt_gross = float(np.dot(opt_ret, opt_w))
            else:
                opt_gross = np.nan

        # --- 基准（universe等权） ---
        bm_rets = []
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        benchmark_ret = np.mean(bm_rets) if bm_rets else np.nan

        turnover_est = 0.5
        cost = turnover_est * (COST_PER_TRADE + STAMP_DUTY)

        records.append({
            "date": month_end,
            "topn":      topn_gross - cost,
            "opt":       opt_gross - cost if pd.notna(opt_gross) else np.nan,
            "benchmark": benchmark_ret,
            "n_opt_holdings": len(w),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计 ──────────────────────────────────────────────

def print_stats(ret: pd.Series, bench: pd.Series, label: str, period: str) -> None:
    common = ret.dropna().index.intersection(bench.dropna().index)
    r, b = ret[common], bench[common]
    if len(r) < 2:
        print(f"\n  [{label} — {period}] 样本不足，跳过")
        return
    excess = r - b
    nav = (1 + r).cumprod()
    print(f"\n  [{label} — {period}]")
    print(f"    样本月数: {len(r)}")
    print(f"    策略年化: {annual_return(nav)*100:.1f}%")
    print(f"    超额年化: {excess.mean()*12*100:+.1f}%")
    print(f"    夏普: {sharpe(r):.3f}")
    print(f"    最大回撤: {max_drawdown(nav)*100:.1f}%")
    print(f"    月胜率vs基准: {(excess > 0).mean()*100:.1f}%")


def print_annual_excess(ret: pd.Series, bench: pd.Series, label: str) -> None:
    common = ret.dropna().index.intersection(bench.dropna().index)
    excess = (ret[common] - bench[common]).dropna()
    print(f"\n  {label} 年度超额（月均）:")
    for y in sorted(excess.index.year.unique()):
        yr = excess[excess.index.year == y]
        print(f"    {y}: {yr.mean()*100:+.2f}%/月  (n={len(yr)})")


def plot_nav(df: pd.DataFrame, output_dir: pathlib.Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    ax_nav, ax_excess = axes
    colors = {"topn": "#60a5fa", "opt": "#f97316"}
    labels = {"topn": "TopN等权（基线）", "opt": "优化组合（风险约束）"}

    for key in ["topn", "opt"]:
        r = df[key].dropna()
        nav = (1 + r).cumprod()
        ax_nav.plot(nav.index, nav.values, color=colors[key], linewidth=1.5, label=labels[key])
        common = df[key].dropna().index.intersection(df["benchmark"].dropna().index)
        excess_cum = (1 + df.loc[common, key] - df.loc[common, "benchmark"]).cumprod()
        ax_excess.plot(excess_cum.index, excess_cum.values, color=colors[key],
                       linewidth=1.5, label=labels[key])

    bm_nav = (1 + df["benchmark"].dropna()).cumprod()
    ax_nav.plot(bm_nav.index, bm_nav.values, color="gray", linewidth=1.2,
               linestyle="--", label="基准（等权）")

    ax_nav.set_title(f"{title} — 净值曲线")
    ax_nav.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_excess.set_title(f"{title} — 超额累积净值")
    ax_excess.axhline(1, color="white", linewidth=0.8, linestyle="--")
    ax_excess.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_excess.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "nav.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"净值图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame, ret_panel: pd.DataFrame,
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

    print(f"\n  回测 TopN等权 vs 优化组合（Top{top_n}）...")
    df = run_backtest_compare(close_panel, ret_panel, members_file,
                              icir_weights, industry_map, top_n)

    if df.empty:
        print("  无有效回测结果")
        return

    df.to_csv(out_dir / "ret_compare.csv")

    print(f"\n{'='*60}")
    print(f"指数：{index_name}  Top{top_n}  TopN等权 vs 风险约束优化组合")
    print(f"{'='*60}")
    print(f"  优化组合平均持仓数: {df['n_opt_holdings'].mean():.1f}")

    for key, label in [("topn", "TopN等权"), ("opt", "优化组合")]:
        ret = df[key]
        bench = df["benchmark"]
        is_mask  = df.index <= IS_END
        oos_mask = df.index >= OOS_START
        print_stats(ret, bench, label, "全样本")
        print_stats(ret[is_mask], bench[is_mask], label, "IS（2016-2024）")
        print_stats(ret[oos_mask], bench[oos_mask], label, "OOS（2024-2026）")

    print_annual_excess(df["topn"], df["benchmark"], "TopN等权")
    print_annual_excess(df["opt"], df["benchmark"], "优化组合")

    plot_nav(df, out_dir, f"{index_name} Top{top_n}")


def main():
    parser = argparse.ArgumentParser(description="风险约束组合优化 V1")
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
    ret_panel = close_panel.pct_change()
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
    print(f"  行业映射：{len(industry_map)} 只")

    print("加载市值快照...")
    load_valuation()
    print()

    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        run_one_index(key, close_panel, ret_panel, args.top, industry_map)

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
