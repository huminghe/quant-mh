"""
商品期货价格动量/持仓变化 → 产业链行业传导因子截面IC验证

来源：审计后新提出候选①（2026-09-09），与已排除的"宏观PPI/CPI行业传导"
（因tushare cn_ppi/cn_cpi颗粒度太粗无法映射行业，见research_index_enhancement.md
第十六轮候选③）不同——本因子直接用商品期货盘面数据（价格/持仓量），不经过
PPI/CPI中转，绕开了粒度问题；与已排除的"产业链行业相关系数异常"（第十七轮
候选②，机制是行业收益相关性均值回归）也不同——本因子机制是原材料成本/景气度
对下游行业的领先传导，非统计层面的相关性回归。

范围说明：本因子只对原材料相关申万一级行业（钢铁/有色金属/煤炭/石油石化/
基础化工/建筑材料）有理论传导机制，这些行业在沪深300成分股中仅42只，低于
50只截面IC最小样本门槛；中证500中83只，刚过门槛。经用户2026-09-09确认，
本次IC检验范围限定中证500。

因子定义（同一行业内所有股票共享同一因子值，即"行业级信号广播到个股"）：
  mom_1m  = 商品期货主力连续合约近1个月收益率
  oi_chg  = 商品期货主力连续合约近1个月持仓量变化率
  两个信号分别做IC检验；若品种有多个（如钢铁=RB+HC+I），取行业内品种
  信号等权平均。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目
既有 factor_ic_*.py 方法论。入选阈值：|IC均值|>=0.03 且年度同向占比>=60%。

用法：
  cd a_stock/backtest
  python factor_ic_futures_commodity.py
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
from fetch_futures_commodity import load_futures_commodity

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_futures_commodity"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50

HS500_MEMBERS_FILE = DATA_DIR / "hs500_members.parquet"
SW_INDUSTRY_FILE   = DATA_DIR / "stock_sw_industry.parquet"

# 目标行业（本次候选①理论传导机制覆盖范围）
TARGET_INDUSTRIES = ["钢铁", "有色金属", "煤炭", "石油石化", "基础化工", "建筑材料"]


# ── 工具函数（同 factor_ic_sue.py）──────────────────────────

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


# ── 商品期货信号计算 ──────────────────────────────────────

def compute_commodity_industry_signals() -> pd.DataFrame:
    """
    按月末计算每个行业的商品信号（mom_1m、oi_chg），行业内多品种等权平均。
    返回：index=月末日期，columns=行业名，两组信号分别放两个DataFrame。
    """
    fut = load_futures_commodity()
    fut = fut.set_index("trade_date").sort_index()

    industries = fut["industry"].unique()
    mom_records, oi_records = {}, {}

    for industry in industries:
        sub = fut[fut["industry"] == industry]
        codes = sub["ts_code"].unique()

        mom_per_code, oi_per_code = {}, {}
        for code in codes:
            s = sub[sub["ts_code"] == code].sort_index()
            monthly = s.resample("ME").last()
            mom = monthly["close"].pct_change(1)
            oi_chg = monthly["oi"].pct_change(1)
            mom_per_code[code] = mom
            oi_per_code[code] = oi_chg

        mom_df = pd.DataFrame(mom_per_code)
        oi_df = pd.DataFrame(oi_per_code)
        mom_records[industry] = mom_df.mean(axis=1)
        oi_records[industry] = oi_df.mean(axis=1)

    mom_panel = pd.DataFrame(mom_records)
    oi_panel = pd.DataFrame(oi_records)
    return mom_panel, oi_panel


def load_stock_industry_map() -> dict:
    df = pd.read_parquet(SW_INDUSTRY_FILE)
    return dict(zip(df["ts_code"], df["sw_industry"]))


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(close_panel: pd.DataFrame, signal_panel: pd.DataFrame,
                        stock_industry: dict) -> pd.DataFrame:
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

        pit_members = load_members_pit(month_end, members_file=HS500_MEMBERS_FILE)
        if not pit_members:
            continue

        # 只保留目标行业成分股
        target_codes = [c for c in pit_members
                         if stock_industry.get(c) in TARGET_INDUSTRIES
                         and c in close_panel.columns]
        if len(target_codes) < MIN_STOCKS_PER_CROSS:
            continue

        # 找信号面板中 <= month_end 的最近一个月度信号
        sig_dates = signal_panel.index[signal_panel.index <= month_end]
        if len(sig_dates) == 0:
            continue
        sig_date = sig_dates[-1]

        factor_vals = {}
        for code in target_codes:
            industry = stock_industry.get(code)
            if industry in signal_panel.columns:
                v = signal_panel.loc[sig_date, industry]
                if pd.notna(v):
                    factor_vals[code] = v
        factor = pd.Series(factor_vals)
        if len(factor) < MIN_STOCKS_PER_CROSS:
            continue

        factor = winsorize(factor)
        factor = standardize(factor)

        close_row = close_panel[target_codes].loc[month_end].dropna()
        fwd_prices_next = close_panel[target_codes].loc[next_month_end].dropna()
        common = close_row.index.intersection(fwd_prices_next.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices_next[common] / close_row[common] - 1
        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({"date": month_end, "ic": ic, "n_stocks": len(common)})

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总（同 factor_ic_sue.py）──────────────────────────

def summarize_ic(ic_series: pd.Series) -> dict:
    clean = ic_series.dropna()
    overall_mean = clean.mean()
    yearly = clean.groupby(clean.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(overall_mean)).mean() if overall_mean != 0 else 0.0
    passed = abs(overall_mean) >= 0.03 and same_sign >= 0.6
    return {
        "样本月数": len(clean),
        "IC均值": round(overall_mean, 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(overall_mean / clean.std(), 3) if clean.std() > 0 else np.nan,
        "IC>0占比": f"{(clean > 0).mean() * 100:.1f}%",
        "年度同向占比": f"{same_sign * 100:.1f}%",
        "通过初筛": passed,
    }


def print_annual_ic(ic_series: pd.Series, label: str) -> None:
    clean = ic_series.dropna()
    yearly = clean.groupby(clean.index.year).mean()
    print(f"\n  {label} 年度IC均值:")
    for y in sorted(yearly.index):
        n = (clean.index.year == y).sum()
        print(f"    {y}: {yearly[y]:+.4f}  (n={n})")


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stock_industry = load_stock_industry_map()

    hs500 = pd.read_parquet(HS500_MEMBERS_FILE)
    all_codes = sorted(hs500["con_code"].unique())
    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"面板大小：{close_panel.shape}")

    print("计算商品期货行业信号...")
    mom_panel, oi_panel = compute_commodity_industry_signals()
    print(f"行业信号覆盖：{list(mom_panel.columns)}\n")

    for sig_name, sig_panel in [("价格动量mom_1m", mom_panel), ("持仓变化oi_chg", oi_panel)]:
        print(f"{'='*60}")
        print(f"信号：{sig_name}（中证500，目标行业：{'/'.join(TARGET_INDUSTRIES)}）")
        print(f"{'='*60}")

        ic_df = compute_monthly_ic(close_panel, sig_panel, stock_industry)
        if ic_df.empty:
            print("  无有效数据")
            continue

        out_dir = OUTPUT_DIR / sig_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ic_df.to_csv(out_dir / "ic_series.csv")

        stats = summarize_ic(ic_df["ic"])
        print(f"  IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")

        print_annual_ic(ic_df["ic"], sig_name)
        plot_ic(ic_df, out_dir, f"中证500商品传导 {sig_name}")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
