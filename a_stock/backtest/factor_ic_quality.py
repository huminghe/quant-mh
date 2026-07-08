"""
质量/估值因子截面 IC 验证（ROE_TTM + EP）
- ROE_TTM：tushare fina_indicator.roe_dt，point-in-time（ann_date）
- EP：EPS（fina_indicator.eps）/ 月末收盘价，point-in-time
- 验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率）
- 支持多指数：沪深300 / 中证500

用法：
  cd a_stock/backtest
  python factor_ic_quality.py               # 默认跑全部
  python factor_ic_quality.py --index hs300
  python factor_ic_quality.py --factor roe  # 只跑 ROE_TTM
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

# 加载数据模块
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR
from fetch_financials import load_financials

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_quality"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50   # 截面最小有效股票数

INDEX_CONFIG = {
    "hs300": {
        "name": "沪深300",
        "members_file": DATA_DIR / "hs300_members.parquet",
    },
    "hs500": {
        "name": "中证500",
        "members_file": DATA_DIR / "hs500_members.parquet",
    },
}

# 要验证的因子
FACTOR_CONFIG = {
    "roe": {
        "name": "ROE_TTM",
        "field": "roe_dt",        # tushare fina_indicator 字段名
        "direction": 1,           # +1 = 越大越好，-1 = 越小越好
        "description": "摊薄净资产收益率（TTM）",
    },
    "ep": {
        "name": "EP（盈利收益率）",
        "field": "eps",           # EPS，需结合价格计算 EP=EPS/price
        "direction": 1,           # 越大越好（低估值高盈利）
        "description": "EPS / 月末收盘价（倒PER）",
    },
}


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


def cross_section_rank_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret"]
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
    return ic


# ── 财务数据缓存（避免重复读取）──────────────────────────

_fina_cache: dict[str, pd.DataFrame] = {}

def get_fina(ts_code: str) -> pd.DataFrame:
    if ts_code not in _fina_cache:
        _fina_cache[ts_code] = load_financials(ts_code)
    return _fina_cache[ts_code]


def get_fina_value_pit(ts_code: str, as_of_date: pd.Timestamp,
                       field: str) -> float | None:
    """point-in-time 取财务字段值：只用 ann_date <= as_of_date 的最新记录"""
    df = get_fina(ts_code)
    if df.empty:
        return None
    valid = df[(df["ann_date"] <= as_of_date) & df[field].notna()]
    if valid.empty:
        return None
    return float(valid.iloc[-1][field])


# ── 因子计算 ──────────────────────────────────────────────

def compute_factor_cross_section(
    codes: list[str],
    month_end: pd.Timestamp,
    factor_key: str,
    close_row: pd.Series,   # 当月末各股收盘价（Series，index=ts_code）
) -> pd.Series:
    """
    计算截面因子值（point-in-time）。
    返回 Series，index=ts_code，值为因子得分。
    """
    cfg = FACTOR_CONFIG[factor_key]
    values = {}

    for code in codes:
        raw = get_fina_value_pit(code, month_end, cfg["field"])
        if raw is None or np.isnan(raw):
            continue

        if factor_key == "ep":
            # EP = EPS / price
            price = close_row.get(code)
            if price is None or np.isnan(price) or price <= 0:
                continue
            values[code] = raw / price
        else:
            values[code] = raw

    if not values:
        return pd.Series(dtype=float)

    s = pd.Series(values)
    # 方向调整：direction=-1 时取负（因子越小越好 → 反向使用）
    s = s * cfg["direction"]
    return s


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    factor_key: str,
    members_file: pathlib.Path,
) -> pd.DataFrame:
    """
    按月末截面计算因子 Rank IC。
    返回 DataFrame，index=月末日期，columns=['ic', 'n_stocks', 'factor_mean']
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

        # point-in-time 成分股
        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        # 当月末收盘价（用于 EP 计算 + 下月收益）
        close_row = close_panel[available].loc[month_end].dropna()

        # 计算截面因子值
        factor = compute_factor_cross_section(
            list(close_row.index), month_end, factor_key, close_row
        )
        if len(factor) < MIN_STOCKS_PER_CROSS:
            continue

        # 去极值 + 标准化
        factor = winsorize(factor)
        factor = standardize(factor)

        # 下月收益率
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common = close_row.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices_next[common] / close_row[common] - 1

        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({
            "date": month_end,
            "ic": ic,
            "n_stocks": len(common),
            "factor_mean": factor.mean(),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    cfg = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    return {
        "因子": cfg["name"],
        "样本月数": len(clean),
        "IC均值": round(clean.mean(), 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(clean.mean() / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
        "|IC|>0.02占比": f"{(clean.abs() > 0.02).mean() * 100:.1f}%",
        "说明": cfg["description"],
    }


# ── 画图 ──────────────────────────────────────────────────

def plot_ic_results(ic_results: dict, output_dir: pathlib.Path,
                    title_prefix: str = "") -> None:
    """ic_results: {factor_key: DataFrame(ic, n_stocks, ...)}"""
    n = len(ic_results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (fkey, df) in enumerate(ic_results.items()):
        ax_bar = axes[row_idx][0]
        ax_cum = axes[row_idx][1]
        ic = df["ic"].dropna()
        fname = FACTOR_CONFIG[fkey]["name"]

        for ax in [ax_bar, ax_cum]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")

        colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
        ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
        ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5,
                       linestyle="-", label=f"均值 {ic.mean():.4f}")
        ax_bar.set_title(f"{title_prefix} {fname} — 月度 Rank IC".strip())
        ax_bar.set_ylabel("Rank IC")
        ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
        ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        cum_ic = ic.cumsum()
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
        ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
        ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_cum.set_title(f"{fname} — 累积 IC（ICIR={icir:.3f}）")
        ax_cum.set_ylabel("累积 Rank IC")
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(index_key: str, close_panel: pd.DataFrame,
                  factor_keys: list[str]) -> pd.DataFrame:
    cfg = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name   = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_results  = {}
    summary_rows = []

    for fkey in factor_keys:
        fname = FACTOR_CONFIG[fkey]["name"]
        print(f"  [{index_name}] {fname}...")
        ic_df = compute_monthly_ic(close_panel, fkey, members_file)
        if ic_df.empty:
            print(f"    无有效数据（请先运行 fetch_financials.py）")
            continue
        ic_results[fkey] = ic_df

        stats = summarize_ic(ic_df["ic"], fkey)
        summary_rows.append(stats)
        print(f"    IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  n={stats['样本月数']}月")

        ic_df.to_csv(out_dir / f"ic_{fkey}.csv")

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).set_index("因子")
    summary_df.to_csv(out_dir / "ic_summary.csv")

    if ic_results:
        plot_ic_results(ic_results, out_dir, title_prefix=index_name)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="质量/估值因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"],
                        default="all")
    parser.add_argument("--factor", choices=list(FACTOR_CONFIG.keys()) + ["all"],
                        default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys  = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]
    factor_keys = list(FACTOR_CONFIG.keys()) if args.factor == "all" else [args.factor]

    # 收集需要的股票代码，一次性加载收盘价面板
    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）\n")

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, factor_keys)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（质量/估值因子 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("解读：")
        print("  IC > 0 = 因子越大下月收益越高（正向因子有效）")
        print("  |ICIR| > 0.3 显著，> 0.5 强显著")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
