"""
限售解禁因子截面IC验证（指数增强另类数据新方向）

因子定义：截面月末时点，未来N个自然日内即将解禁（且已公告，ann_date<=月末）
的股份占总股本比例之和（float_ratio汇总）。
逻辑：解禁临近→减持预期升温→压制股价，因子方向为负（解禁比例越高，
下月收益预期越低）。这与已证伪的基本面景气度因子（净利润增速/ROE/
现金流质量/杠杆/营收增速）逻辑完全不同，是筹码结构类信号，此前从未
在此IC框架下测试。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率）。
数据源：share_float.parquet（fetch_share_float.py下载，2016-2026，
覆盖4716只股票，float_ratio缺失率2.3%）。

用法：
  cd a_stock/backtest
  python factor_ic_share_float.py --index hs500
  python factor_ic_share_float.py --index hs500 --window 60
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

SHARE_FLOAT_FILE = DATA_DIR / "share_float.parquet"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_share_float"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50

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


# ── 工具函数 ──────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
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


def load_share_float() -> pd.DataFrame:
    df = pd.read_parquet(SHARE_FLOAT_FILE)
    return df


def compute_upcoming_float_ratio(
    sf: pd.DataFrame, codes: list[str], month_end: pd.Timestamp, window_days: int
) -> pd.Series:
    """
    截面因子：未来window_days自然日内即将解禁（且已公告）的股份占总股本比例之和。
    只用ann_date<=month_end的记录（PIT：不能用尚未公告的解禁信息）。

    没有解禁事件的股票记为0（无解禁压力，正常状态），而非排除在外——
    多数股票在任意时点都没有即将解禁的事件，这不是数据缺失，是因子取值
    本身的常态。若排除会导致截面覆盖率虚低（此前bug：把该记0的股票当NaN丢弃）。
    """
    window_end = month_end + pd.Timedelta(days=window_days)
    mask = (
        sf["ts_code"].isin(codes)
        & (sf["ann_date"] <= month_end)
        & (sf["float_date"] > month_end)
        & (sf["float_date"] <= window_end)
    )
    sub = sf[mask]
    agg = sub.groupby("ts_code")["float_ratio"].sum() if not sub.empty else pd.Series(dtype=float)
    return agg.reindex(codes, fill_value=0.0)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    sf: pd.DataFrame,
    members_file: pathlib.Path,
    window_days: int,
) -> pd.DataFrame:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        month_end = pd.Timestamp(month_end)
        next_month_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        factor = compute_upcoming_float_ratio(sf, available, month_end, window_days)
        if len(factor) < MIN_STOCKS_PER_CROSS // 2:
            continue

        # 方向：解禁比例越高 → 未来收益预期越低 → 取负后正向使用
        factor = -factor
        factor = winsorize(factor)
        factor = standardize(factor)

        close_row = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common = close_row.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS // 2:
            continue

        fwd_ret = fwd_prices_next[common] / close_row[common] - 1
        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({
            "date": month_end,
            "ic": ic,
            "n_stocks": len(common),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def summarize_ic(ic_series: pd.Series) -> dict:
    clean = ic_series.dropna()
    return {
        "样本月数": len(clean),
        "IC均值": round(clean.mean(), 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(clean.mean() / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
        "|IC|>0.02占比": f"{(clean.abs() > 0.02).mean() * 100:.1f}%",
    }


def print_annual_ic(ic_series: pd.Series, label: str) -> None:
    clean = ic_series.dropna()
    print(f"\n  {label} 年度IC均值:")
    for y in sorted(clean.index.year.unique()):
        yr = clean[clean.index.year == y]
        print(f"    {y}: {yr.mean():+.4f}  (n={len(yr)})")


def plot_ic(ic_df: pd.DataFrame, output_dir: pathlib.Path, title: str) -> None:
    ic = ic_df["ic"].dropna()
    fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(16, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax_bar, ax_cum]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
    ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
    ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5, label=f"均值 {ic.mean():.4f}")
    ax_bar.set_title(f"{title} — 月度 Rank IC")
    ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    cum_ic = ic.cumsum()
    icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    ax_cum.plot(cum_ic.index, cum_ic.values, color="#60a5fa", linewidth=1.5)
    ax_cum.fill_between(cum_ic.index, cum_ic.values, alpha=0.2, color="#60a5fa")
    ax_cum.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax_cum.set_title(f"累积 IC（ICIR={icir:.3f}）")
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out_path = output_dir / "ic_series.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="限售解禁因子截面IC验证")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="all")
    parser.add_argument("--window", type=int, default=30, help="未来解禁窗口天数")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    print("加载限售解禁数据...")
    sf = load_share_float()
    print(f"  共 {len(sf)} 行，覆盖 {sf['ts_code'].nunique()} 只股票\n")

    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}\n")

    for key in index_keys:
        cfg = INDEX_CONFIG[key]
        name = cfg["name"]
        members_file = cfg["members_file"]
        if not members_file.exists():
            print(f"跳过 {name}：成分股快照不存在")
            continue

        out_dir = OUTPUT_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"指数：{name}（{key}）  解禁窗口={args.window}天")
        print(f"{'='*60}")

        ic_df = compute_monthly_ic(close_panel, sf, members_file, args.window)
        if ic_df.empty:
            print("  无有效数据")
            continue

        ic_df.to_csv(out_dir / "ic_series.csv")
        stats = summarize_ic(ic_df["ic"])
        print(f"  IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  n={stats['样本月数']}月")

        print_annual_ic(ic_df["ic"], name)
        plot_ic(ic_df, out_dir, f"{name} 解禁因子（{args.window}天窗口）")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
