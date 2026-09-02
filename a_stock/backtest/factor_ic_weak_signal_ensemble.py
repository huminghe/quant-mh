"""
ML弱信号集成迁移前提检验：拥挤度(crowding) + 成交量确认(vol_ratio)

背景：ETF轮动侧`etf_rotation_v16_signal_combo_ablation.py`验证过三个弱信号
（拥挤度/成交量确认/资金流反向）等权集成后对45只手工精选ETF池有正贡献，
但交叉验证显示该集成信号仅对结构清晰的小池子有效，换到431只机械化候选池
后反而负贡献（详见 a_stock/docs/research.md）。指数增强的成分股池（沪深300/
中证500，几百只，无清晰分组）结构与两者都不同，且指数增强的打分排序类
框架历史上已40+次失败，接入前先测最低成本的前提：这两个信号在成分股层面
是否有独立预测力（IC初筛）。

本脚本只跳过flow信号（ETF份额变化率在个股层面无直接对应物，是否需要找
替代定义留待crowding/vol_ratio结果出来后再决定），不做组合回测/消融，
前提不成立就止步于此。

信号定义（直接复用ETF轮动侧`etf_rotation_v16_signal_combo_ablation.py`
已验证的计算逻辑，不重新设计）：
- 拥挤度crowding：收益率60日滚动相关系数矩阵，对角线置NaN后按行均值得
  每日平均相关性，按252日历史百分位反转（越不拥挤越好，direction=+1
  因子构造时已内含反转，此处不再额外乘方向系数）。
- 成交量确认vol_ratio：amount 5日均值/20日均值，放量确认越强越好
  （direction=+1，正向使用，不额外反转）。

验证方式：月度截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目既有
factor_ic_*.py方法论（沪深300/中证500成分股，月度截面，PIT成分股）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%（项目既定阈值）。

用法：
  cd a_stock/backtest
  python factor_ic_weak_signal_ensemble.py               # 默认跑全部
  python factor_ic_weak_signal_ensemble.py --index hs300
"""

import sys
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR

STOCK_DIR = DATA_DIR / "stock_daily"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_weak_signal_ensemble"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50

CORR_WINDOW = 60
CORR_HIST_WINDOW = 252
VOL_RATIO_SHORT = 5
VOL_RATIO_LONG = 20

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

FACTOR_CONFIG = {
    "crowding":  {"name": "拥挤度crowding（60日相关性252日历史反转）", "direction": +1},
    "vol_ratio": {"name": "成交量确认vol_ratio（5日/20日均量比）",      "direction": +1},
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


# ── 数据加载 ──────────────────────────────────────────────

def load_amount_panel(codes: list[str]) -> pd.DataFrame:
    """读取成交额面板（宽格式，index=trade_date，columns=ts_code），字段取自 stock_daily"""
    frames = {}
    for code in codes:
        path = STOCK_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "amount"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames[code] = df.set_index("trade_date")["amount"]
    if not frames:
        raise FileNotFoundError("没有找到任何 amount 数据")
    return pd.DataFrame(frames).sort_index()


# ── 因子计算（日频面板） ────────────────────────────────────

def compute_crowding_daily(close_panel: pd.DataFrame, corr_window: int = CORR_WINDOW,
                            hist_window: int = CORR_HIST_WINDOW) -> pd.DataFrame:
    """
    拥挤度：60日滚动收益率相关系数矩阵，对角线置NaN后按行均值得当日平均相关性，
    再按252日历史百分位反转（越不拥挤即当前相关性低于历史越多越好）。
    直接复用 etf_rotation_v16_signal_combo_ablation.py 的 calc_crowding 逻辑，
    仅将 ETF 收盘价面板换成个股收盘价面板。
    """
    codes = list(close_panel.columns)
    ret = close_panel.pct_change()
    ret_values = ret.values  # (n, k)，用numpy向量化替代逐日pandas操作
    n, k = ret_values.shape
    crowding_raw = np.full((n, k), np.nan)

    for i in range(corr_window, n):
        win = ret_values[i - corr_window: i]  # (corr_window, k)
        valid_cols = ~np.isnan(win).any(axis=0)
        if valid_cols.sum() < 5:
            continue
        sub = win[:, valid_cols]
        corr_arr = np.corrcoef(sub, rowvar=False)  # numpy比pandas .corr() 少一层索引开销
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = np.nanmean(corr_arr, axis=1)
        crowding_raw[i, valid_cols] = avg_corr

    crowding_pct = np.full((n, k), np.nan)
    for i in range(hist_window + corr_window, n):
        hist = crowding_raw[i - hist_window: i]  # (hist_window, k)
        curr = crowding_raw[i]  # (k,)
        valid_count = np.sum(~np.isnan(hist), axis=0)
        # nan < x 在numpy中恒为False，无需单独屏蔽nan位，用valid_count控制分母和有效性
        less_count = np.sum(hist < curr[np.newaxis, :], axis=0)
        pct = np.divide(less_count, valid_count, out=np.full(k, np.nan), where=valid_count > 0)
        pct[(valid_count < 20) | np.isnan(curr)] = np.nan
        crowding_pct[i] = pct

    return pd.DataFrame(crowding_pct, index=close_panel.index, columns=codes)


def compute_vol_ratio_daily(amount_panel: pd.DataFrame, short: int = VOL_RATIO_SHORT,
                             long: int = VOL_RATIO_LONG) -> pd.DataFrame:
    """成交量确认：amount 短窗口均值 / 长窗口均值，放量确认越强越好"""
    return amount_panel.rolling(short).mean() / amount_panel.rolling(long).mean().replace(0, np.nan)


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    factor_panels: dict[str, pd.DataFrame],
    members_file: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    """
    月度截面 Rank IC。两个因子面板均已是逐日算好的最终因子值（crowding/vol_ratio
    的滚动窗口已在 compute_*_daily 里处理完），月末截面直接取值，不再额外滚动均值
    （区别于 factor_ic_liquidity.py 里对原始字段做 window 均值的写法）。
    """
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = {fkey: [] for fkey in factor_panels}

    for i, month_end in enumerate(monthly_last[:-1]):
        month_end = pd.Timestamp(month_end)
        next_month_end = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        close_row = close_panel[available].loc[month_end].dropna()
        fwd_prices_next = close_panel[available].loc[next_month_end].dropna()
        common_ret = close_row.index.intersection(fwd_prices_next.index)
        if len(common_ret) < MIN_STOCKS_PER_CROSS:
            continue
        fwd_ret = fwd_prices_next[common_ret] / close_row[common_ret] - 1

        for fkey, panel in factor_panels.items():
            cfg = FACTOR_CONFIG[fkey]
            cols = [c for c in available if c in panel.columns]
            if not cols or month_end not in panel.index:
                continue
            factor = panel[cols].loc[month_end].dropna()
            if len(factor) < MIN_STOCKS_PER_CROSS:
                continue

            factor = factor * cfg["direction"]
            factor = winsorize(factor)
            factor = standardize(factor)

            common = factor.index.intersection(common_ret)
            if len(common) < MIN_STOCKS_PER_CROSS:
                continue

            ic = cross_section_rank_ic(factor[common], fwd_ret[common])
            records[fkey].append({"date": month_end, "ic": ic, "n_stocks": len(common)})

    return {
        fkey: (pd.DataFrame(recs).set_index("date") if recs else pd.DataFrame())
        for fkey, recs in records.items()
    }


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    """
    年度同向占比：按项目既定口径，是"各年度IC均值与全样本IC均值符号一致"
    的年份占比，不是月度符号占比。
    """
    cfg = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    overall_mean = clean.mean()
    yearly = clean.groupby(clean.index.year).mean()
    same_sign = (np.sign(yearly) == np.sign(overall_mean)).mean() if overall_mean != 0 else 0.0
    passed = abs(overall_mean) >= 0.03 and same_sign >= 0.6
    return {
        "因子": cfg["name"],
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


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(
    index_key: str,
    close_panel: pd.DataFrame,
    factor_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cfg = INDEX_CONFIG[index_key]
    members_file = cfg["members_file"]
    index_name = cfg["name"]

    if not members_file.exists():
        print(f"跳过 {index_name}：成分股快照不存在")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{index_name}] 计算月度截面IC...")
    ic_results = compute_monthly_ic(close_panel, factor_panels, members_file)

    summary_rows = []
    for fkey, ic_df in ic_results.items():
        if ic_df.empty:
            print(f"    {FACTOR_CONFIG[fkey]['name']}：无有效数据")
            continue
        stats = summarize_ic(ic_df["ic"], fkey)
        summary_rows.append(stats)
        print(f"    {stats['因子']:<40} IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")
        print_annual_ic(ic_df["ic"], stats["因子"])
        ic_df.to_csv(out_dir / f"ic_{fkey}.csv")

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).set_index("因子")
    summary_df.to_csv(out_dir / "ic_summary.csv")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="ML弱信号集成迁移前提检验（拥挤度+成交量确认）")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_keys = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]

    all_codes = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())
    all_codes = list(all_codes)

    print(f"加载收盘价/成交额面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=all_codes)
    amount_panel = load_amount_panel(all_codes)
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    print("计算拥挤度信号（60日相关性矩阵，252日历史百分位，计算量较大，请耐心等待）...")
    crowding_panel = compute_crowding_daily(close_panel)

    print("计算成交量确认信号（5日/20日均量比）...")
    vol_ratio_panel = compute_vol_ratio_daily(amount_panel)

    factor_panels = {
        "crowding": crowding_panel,
        "vol_ratio": vol_ratio_panel,
    }

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"\n{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, factor_panels)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（ML弱信号迁移前提检验 Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("判定标准：|IC均值|>=0.03 且年度同向占比>=60% 为通过初筛")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
