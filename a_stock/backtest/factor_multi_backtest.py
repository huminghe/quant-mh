"""
多因子选股回测（指数增强）
- 因子：特质波动率（低波动）+ EP（估值）+ ROE_TTM（质量）+ 反转（价格）
- 合成方式：ICIR 加权 vs 等权对比
- 选股：Top N 等权，月度调仓
- 验证：IS（前80%）/ OOS（后20%）
- 基准：指数买持（沪深300 or 中证500，用成分股等权代理）

用法：
  cd a_stock/backtest
  python factor_multi_backtest.py               # 沪深300 + 中证500 全跑
  python factor_multi_backtest.py --index hs300 --top 30
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

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_multi_backtest"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_CROSS = 50   # 截面最小有效股票数
MIN_OBS_VOL      = 30   # 波动率计算最小观测数

# IS/OOS 分割（按时间，非随机）
IS_END   = "2024-01-31"   # 前80%约 2016-2024
OOS_START = "2024-02-01"  # 后20%约 2024-2026

# 回测成本（单次调仓）：佣金万1双边 + 滑点万2双边
COST_PER_TRADE = 0.0030   # 买入 0.15% + 卖出 0.15%（ETF 无印花税，股票需调整）
STAMP_DUTY     = 0.001    # 印花税，仅卖出收

RISK_FREE_ANNUAL = 0.02   # 无风险利率

INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
        "factor_icir": {
            "ep":       0.150,   # EP，方向=+1
            "roe":      0.196,   # ROE_TTM，方向=+1
            "reversal": 0.101,   # 反转（5日），方向=-1（跌多=好）
        },
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
        "factor_icir": {
            "ep":       0.255,
            "roe":      0.195,
            "reversal": 0.123,   # 63日反转
        },
    },
}

# 因子参数
IVOL_WINDOW     = 63    # 特质波动率回望窗口
REVERSAL_WINDOW_HS300 = 5   # 沪深300 反转窗口（5日最强）
REVERSAL_WINDOW_HS500 = 63  # 中证500 反转窗口（63日最强）


# ── 工具函数 ──────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lo, hi)


def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def sharpe(ret: pd.Series, freq: int = 12) -> float:
    """年化夏普，ret 为月度收益率序列"""
    if len(ret) < 2:
        return np.nan
    ann = ret.mean() * freq
    std = ret.std() * np.sqrt(freq)
    if std < 1e-8:
        return np.nan
    return (ann - RISK_FREE_ANNUAL) / std


def max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    return dd.min()


def annual_return(nav: pd.Series, freq: int = 12) -> float:
    n = len(nav)
    if n < 2:
        return np.nan
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = n / freq
    return (1 + total) ** (1 / years) - 1


# ── 财务数据缓存 ──────────────────────────────────────────

_fina_cache: dict[str, pd.DataFrame] = {}

def get_fina(ts_code: str) -> pd.DataFrame:
    if ts_code not in _fina_cache:
        _fina_cache[ts_code] = load_financials(ts_code)
    return _fina_cache[ts_code]


def get_fina_pit(ts_code: str, as_of: pd.Timestamp, field: str) -> float | None:
    df = get_fina(ts_code)
    if df.empty:
        return None
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    if valid.empty:
        return None
    return float(valid.iloc[-1][field])


# ── 因子计算 ──────────────────────────────────────────────

def compute_ivol(close_panel: pd.DataFrame, codes: list[str],
                 month_end: pd.Timestamp) -> pd.Series:
    """特质波动率（LOO 市场代理，取负使低波动=高得分）"""
    hist = close_panel[codes].pct_change().loc[:month_end].iloc[-(IVOL_WINDOW + 1):-1]
    if len(hist) < MIN_OBS_VOL:
        return pd.Series(dtype=float)

    n_valid_per_day = hist.notna().sum(axis=1)
    col_sum = hist.sum(axis=1)
    ivol = {}
    for code in codes:
        sr = hist[code].dropna()
        if len(sr) < MIN_OBS_VOL:
            continue
        n = n_valid_per_day.loc[sr.index]
        loo = (col_sum.loc[sr.index] - sr) / (n - 1).clip(lower=1)
        X = np.column_stack([np.ones(len(loo)), loo.values])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, sr.values, rcond=None)
            resid = sr.values - X @ beta
            ivol[code] = resid.std() * np.sqrt(252)
        except Exception:
            continue
    s = pd.Series(ivol)
    return -s   # 取负：低波动 → 高得分


def compute_reversal(close_panel: pd.DataFrame, codes: list[str],
                     month_end: pd.Timestamp, window: int) -> pd.Series:
    """价格反转因子：过去 window 日累积收益取负"""
    hist = close_panel[codes].loc[:month_end]
    if len(hist) < window + 1:
        return pd.Series(dtype=float)
    ret = hist.iloc[-1] / hist.iloc[-(window + 1)] - 1
    return -ret.dropna()  # 取负：跌多 → 高得分


def compute_ep(codes: list[str], month_end: pd.Timestamp,
               close_row: pd.Series) -> pd.Series:
    """EP = EPS / 价格，point-in-time"""
    ep = {}
    for code in codes:
        eps = get_fina_pit(code, month_end, "eps")
        price = close_row.get(code)
        if eps is None or price is None or np.isnan(price) or price <= 0:
            continue
        val = eps / price
        if not np.isnan(val):
            ep[code] = val
    return pd.Series(ep)


def compute_roe(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """ROE_TTM，point-in-time"""
    roe = {}
    for code in codes:
        val = get_fina_pit(code, month_end, "roe_dt")
        if val is not None and not np.isnan(val):
            roe[code] = val
    return pd.Series(roe)


# ── 截面打分合成 ──────────────────────────────────────────

def compute_composite_score(
    close_panel: pd.DataFrame,
    codes: list[str],
    month_end: pd.Timestamp,
    reversal_window: int,
    icir_weights: dict[str, float],
    weight_scheme: str = "icir",   # "icir" or "equal"
) -> pd.Series:
    """
    计算截面综合得分。
    每个因子截面标准化后，按 ICIR 或等权加权。
    返回 Series，index=ts_code，值越大越好。
    """
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    factors = {
        "reversal": compute_reversal(close_panel, available, month_end, reversal_window),
        "ep":       compute_ep(available, month_end, close_row),
        "roe":      compute_roe(available, month_end),
    }

    # 截面标准化
    norm_factors = {}
    for fname, fs in factors.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm_factors[fname] = fs

    if not norm_factors:
        return pd.Series(dtype=float)

    # 求公共股票集合
    common = None
    for fs in norm_factors.values():
        if common is None:
            common = set(fs.index)
        else:
            common &= set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return pd.Series(dtype=float)
    common = list(common)

    # 加权合成
    if weight_scheme == "icir":
        total_w = sum(icir_weights.get(f, 0) for f in norm_factors)
        if total_w < 1e-8:
            return pd.Series(dtype=float)
        score = pd.Series(0.0, index=common)
        for fname, fs in norm_factors.items():
            w = icir_weights.get(fname, 0) / total_w
            score += fs[common] * w
    else:  # equal
        score = pd.Series(0.0, index=common)
        for fs in norm_factors.values():
            score += fs[common] / len(norm_factors)

    return score.dropna()


# ── 月度回测 ──────────────────────────────────────────────

def run_backtest(
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    reversal_window: int,
    icir_weights: dict[str, float],
    top_n: int = 30,
    weight_scheme: str = "icir",
) -> pd.DataFrame:
    """
    月度调仓选股回测。
    返回 DataFrame，index=月末日期，columns=['strategy', 'benchmark', 'n_stocks']
    """
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_month_end = monthly_last[i + 1]
        month_end = pd.Timestamp(month_end)
        next_month_end = pd.Timestamp(next_month_end)

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        # 综合得分
        score = compute_composite_score(
            close_panel, available, month_end,
            reversal_window, icir_weights, weight_scheme
        )
        if len(score) < top_n:
            continue

        # 选 Top N
        selected = score.nlargest(top_n).index.tolist()

        # 计算下月收益（等权）
        ret_list = []
        for code in selected:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_month_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret_list.append(p1 / p0 - 1)

        if not ret_list:
            continue

        # 扣交易成本（假设每月完全换仓最坏情况）
        gross_ret = np.mean(ret_list)
        # 简化：每月换手率约 50%，每手买+卖成本 0.30% + 印花税 0.10%
        turnover_est = 0.5
        cost = turnover_est * (COST_PER_TRADE + STAMP_DUTY)
        strategy_ret = gross_ret - cost

        # 基准：成分股等权（不扣成本）
        bm_rets = []
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_month_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                bm_rets.append(p1 / p0 - 1)
        benchmark_ret = np.mean(bm_rets) if bm_rets else np.nan

        records.append({
            "date":          month_end,
            "strategy":      strategy_ret,
            "benchmark":     benchmark_ret,
            "n_stocks":      len(ret_list),
            "gross_ret":     gross_ret,
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 绩效统计 ──────────────────────────────────────────────

def print_stats(ret_df: pd.DataFrame, label: str, period: str = "全样本") -> dict:
    strat = ret_df["strategy"].dropna()
    bench = ret_df["benchmark"].dropna()
    common_idx = strat.index.intersection(bench.index)
    strat = strat[common_idx]
    bench = bench[common_idx]
    excess = strat - bench

    nav_s = (1 + strat).cumprod()
    nav_b = (1 + bench).cumprod()

    stats = {
        "期间":          period,
        "样本月数":       len(strat),
        "策略年化":       f"{annual_return(nav_s)*100:.1f}%",
        "基准年化":       f"{annual_return(nav_b)*100:.1f}%",
        "超额年化":       f"{excess.mean()*12*100:.1f}%",
        "策略夏普":       f"{sharpe(strat):.3f}",
        "基准夏普":       f"{sharpe(bench):.3f}",
        "策略最大回撤":   f"{max_drawdown(nav_s)*100:.1f}%",
        "基准最大回撤":   f"{max_drawdown(nav_b)*100:.1f}%",
        "月胜率vs基准":   f"{(excess > 0).mean()*100:.1f}%",
    }
    print(f"\n  [{label} — {period}]")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    return stats


# ── 画图 ──────────────────────────────────────────────────

def plot_nav(results: dict, output_dir: pathlib.Path, title: str) -> None:
    """results: {label: ret_df}"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    colors = ["#60a5fa", "#f97316", "#34d399", "#facc15"]

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    ax_nav, ax_excess = axes

    benchmark_plotted = False
    for (label, ret_df), color in zip(results.items(), colors):
        strat = ret_df["strategy"].dropna()
        bench = ret_df["benchmark"].dropna()
        nav_s = (1 + strat).cumprod()
        nav_b = (1 + bench).cumprod()
        excess_cum = ((1 + strat - bench)).cumprod()

        ax_nav.plot(nav_s.index, nav_s.values, color=color,
                    linewidth=1.5, label=label)
        if not benchmark_plotted:
            ax_nav.plot(nav_b.index, nav_b.values, color="gray",
                        linewidth=1.2, linestyle="--", label="基准（等权）")
            benchmark_plotted = True
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
                  top_n: int) -> None:
    cfg          = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]
    icir_weights = cfg["factor_icir"]
    reversal_win = REVERSAL_WINDOW_HS300 if index_key == "hs300" else REVERSAL_WINDOW_HS500

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  回测 ICIR 加权（Top{top_n}）...")
    ret_icir  = run_backtest(close_panel, members_file, reversal_win,
                             icir_weights, top_n, "icir")

    print(f"  回测等权合成（Top{top_n}）...")
    ret_equal = run_backtest(close_panel, members_file, reversal_win,
                             icir_weights, top_n, "equal")

    if ret_icir.empty:
        print("  无有效回测结果")
        return

    # 保存
    ret_icir.to_csv(out_dir / "ret_icir.csv")
    ret_equal.to_csv(out_dir / "ret_equal.csv")

    # 全样本统计
    print(f"\n{'='*60}")
    print(f"指数：{index_name}  Top{top_n}  反转窗口:{reversal_win}日")
    print(f"{'='*60}")

    for label, ret_df in [("ICIR加权", ret_icir), ("等权合成", ret_equal)]:
        is_df  = ret_df[ret_df.index <= IS_END]
        oos_df = ret_df[ret_df.index >= OOS_START]
        print_stats(ret_df,  label, "全样本")
        print_stats(is_df,   label, "IS（2016-2024）")
        print_stats(oos_df,  label, "OOS（2024-2026）")

    # 画图
    plot_nav({"ICIR加权": ret_icir, "等权合成": ret_equal},
             out_dir, f"{index_name} Top{top_n}")

    # 顺便测试不同 TopN
    print(f"\n  TopN 敏感性（ICIR加权）...")
    for n in [10, 20, 30, 50]:
        if n == top_n:
            continue
        r = run_backtest(close_panel, members_file, reversal_win,
                         icir_weights, n, "icir")
        if r.empty:
            continue
        s = sharpe(r["strategy"].dropna())
        mdd = max_drawdown((1 + r["strategy"].dropna()).cumprod())
        ann = annual_return((1 + r["strategy"].dropna()).cumprod())
        print(f"    Top{n}: 年化={ann*100:.1f}%  夏普={s:.3f}  最大回撤={mdd*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="多因子选股回测")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="all")
    parser.add_argument("--top",   type=int, default=30, help="持仓数量（默认30）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    # 一次性加载所有需要的股票面板
    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")
    print(f"预加载财务数据...")

    # 预热财务缓存（避免逐月重复读取）
    fin_codes = list(all_codes)
    for i, code in enumerate(fin_codes):
        get_fina(code)
        if (i + 1) % 200 == 0:
            print(f"  财务缓存：{i+1}/{len(fin_codes)}")

    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"\n{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        run_one_index(key, close_panel, args.top)

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
