"""
可转债双低因子截面IC验证（Phase 1）

用于验证"双低值=转债价格+100×转股溢价率(%)"这一经典因子在2022年可转债
交易新规前后是否仍然有效。详见 a_stock/docs/research_convertible_bond.md。

数据来源：akshare bond_zh_cov_value_analysis 逐日历史序列
（a_stock/data/cb_premium_history.parquet，已核实覆盖已退市标的，无幸存
者偏差），字段：close/conv_premium_pct。转债标的池取 cb_universe.parquet
中 cb_type=='CB' 的全部1138只（含已退市），不限定沪深300/中证500成分股
（可转债本身不是指数成分概念，用全市场存续池）。

方法：
- 双低值 = close + 100 * conv_premium_pct，月末截面，Top 20%（双低值越小
  越优先）等权分组，T+1建仓（下月首个交易日）
- 月度收益 = 分组内个券月度收益率等权平均（转债无需复权，票面100元发行，
  无分红拆股事件）
- 核心输出：2022年新规前（<2022-01-01）vs 新规后两段的月度收益、Rank IC
  分别统计，判断超额是否在新规后明显收窄

用法：
  cd a_stock/backtest
  python factor_ic_cb_lowprice.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
UNIVERSE_FILE = DATA_DIR / "cb_universe.parquet"
PREMIUM_FILE = DATA_DIR / "cb_premium_history.parquet"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_cb_lowprice"

START_DATE = "2018-01-01"
END_DATE = "2026-08-31"
NEW_RULE_DATE = pd.Timestamp("2022-01-01")  # 沪深交易所可转债新规实施日（2022-01-01起）

MIN_BONDS_PER_CROSS = 30  # 转债市场早期规模小，用比股票IC更低的门槛
TOP_PCT = 0.2


# ── 数据加载 ──────────────────────────────────────────────

def load_premium_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (close_panel, low_score_panel)，宽格式，index=trade_date，columns=ts_code"""
    ph = pd.read_parquet(PREMIUM_FILE)
    ph = ph.dropna(subset=["close", "conv_premium_pct"]).copy()
    ph["low_score"] = ph["close"] + 100 * ph["conv_premium_pct"]

    close_panel = ph.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    low_panel = ph.pivot(index="trade_date", columns="ts_code", values="low_score").sort_index()
    return close_panel, low_panel


# ── IC 与分组收益计算 ─────────────────────────────────────

def cross_section_rank_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret"]
    if len(aligned) < MIN_BONDS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
    return ic


def get_month_ends(close_panel: pd.DataFrame) -> np.ndarray:
    sub = close_panel.loc[START_DATE:END_DATE]
    nat_month_ends = sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        sub.index[sub.index <= m][-1]
        for m in nat_month_ends
        if len(sub.index[sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return monthly_last


def compute_monthly_ic_and_group_ret(
    close_panel: pd.DataFrame, low_panel: pd.DataFrame
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
        # 剔除单月涨跌幅超过100%的异常点（停牌恢复/数据噪声，非正常交易结果）
        fwd_ret = fwd_ret[fwd_ret.abs() < 1.0]

        if month_end not in low_panel.index:
            continue
        factor = low_panel.loc[month_end].dropna()

        common = factor.index.intersection(fwd_ret.index)
        if len(common) < MIN_BONDS_PER_CROSS:
            continue

        ic = cross_section_rank_ic(factor[common], fwd_ret[common])

        # Top 20%等权分组（双低值越小越优先，故取最小的TOP_PCT）
        n_select = max(1, int(len(common) * TOP_PCT))
        selected = factor[common].nsmallest(n_select).index
        group_ret = fwd_ret[selected].mean()
        bench_ret = fwd_ret[common].mean()

        records.append({
            "date": month_end,
            "ic": ic,
            "n_bonds": len(common),
            "group_ret": group_ret,
            "bench_ret": bench_ret,
            "n_selected": len(selected),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series) -> dict:
    clean = ic_series.dropna()
    if clean.empty:
        return {}
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


def summarize_group_ret(df: pd.DataFrame) -> dict:
    excess = (df["group_ret"] - df["bench_ret"]).dropna()
    if excess.empty:
        return {}
    ann_excess = excess.mean() * 12
    return {
        "样本月数": len(excess),
        "月均超额收益": f"{excess.mean()*100:+.3f}%",
        "年化超额收益(gross)": f"{ann_excess*100:+.2f}%",
        "月胜率(超额>0)": f"{(excess > 0).mean()*100:.1f}%",
    }


def print_period_report(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"\n  [{label}] 无有效数据")
        return
    ic_stats = summarize_ic(df["ic"])
    ret_stats = summarize_group_ret(df)
    print(f"\n  [{label}]（样本区间 {df.index[0].date()} ~ {df.index[-1].date()}，n={len(df)}月）")
    if ic_stats:
        print(f"    IC均值={ic_stats['IC均值']:+.4f}  ICIR={ic_stats['ICIR']:+.3f}  "
              f"IC>0占比={ic_stats['IC>0占比']}  年度同向占比={ic_stats['年度同向占比']}  "
              f"{'通过初筛' if ic_stats['通过初筛'] else '未达阈值'}")
    if ret_stats:
        print(f"    Top20%双低组 月均超额={ret_stats['月均超额收益']}  "
              f"年化超额(gross)={ret_stats['年化超额收益(gross)']}  "
              f"月胜率={ret_stats['月胜率(超额>0)']}")


def print_annual(df: pd.DataFrame) -> None:
    print("\n  年度拆分（IC均值 / Top20%组月均超额）:")
    for y in sorted(df.index.year.unique()):
        yr = df[df.index.year == y]
        ic_mean = yr["ic"].mean()
        excess_mean = (yr["group_ret"] - yr["bench_ret"]).mean()
        print(f"    {y}: IC={ic_mean:+.4f}  超额={excess_mean*100:+.3f}%/月  (n={len(yr)})")


# ── 主流程 ────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载双低因子面板（akshare历史序列，含已退市转债）...")
    close_panel, low_panel = load_premium_panel()
    print(f"面板大小：{close_panel.shape}（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    print("\n计算月度截面IC + Top20%分组收益...")
    df = compute_monthly_ic_and_group_ret(close_panel, low_panel)
    if df.empty:
        print("无有效数据，退出")
        return
    df.to_csv(OUTPUT_DIR / "monthly_ic_and_group_ret.csv")

    print(f"\n{'='*70}")
    print("全样本（2018-2026）")
    print(f"{'='*70}")
    print_period_report(df, "全样本")
    print_annual(df)

    pre_rule = df[df.index < NEW_RULE_DATE]
    post_rule = df[df.index >= NEW_RULE_DATE]

    print(f"\n{'='*70}")
    print("新规前后对比（2022-01-01 沪深交易所可转债新规实施）")
    print(f"{'='*70}")
    print_period_report(pre_rule, "新规前 2018-2021")
    print_period_report(post_rule, "新规后 2022-2026")

    summary = pd.DataFrame({
        "全样本": {**summarize_ic(df["ic"]), **summarize_group_ret(df)},
        "新规前2018-2021": {**summarize_ic(pre_rule["ic"]), **summarize_group_ret(pre_rule)},
        "新规后2022-2026": {**summarize_ic(post_rule["ic"]), **summarize_group_ret(post_rule)},
    }).T
    summary.to_csv(OUTPUT_DIR / "period_summary.csv")
    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
