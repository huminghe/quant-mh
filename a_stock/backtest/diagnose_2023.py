"""
诊断 V2 回测 2023 年大幅跑输（-1.70%/月）的原因

检查三个假设：
1. 公共股票集数量（5个因子都有数据的交集）2023年是否大幅缩水
2. 选出的 Top30 行业分布是否异常集中（2023年表现差的行业）
3. 各因子2023年截面 IC 是否出现方向分歧或异常

用法：
  cd a_stock/backtest
  python diagnose_2023.py
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    MIN_STOCKS_CROSS, MIN_STOCKS_SECTOR, MIN_HISTORY_QTRS,
    INDEX_CONFIG,
)

MEMBERS_FILE = INDEX_CONFIG["hs500"]["members_file"]
ICIR_WEIGHTS = INDEX_CONFIG["hs500"]["factor_icir"]

# ── 诊断区间：关注 2022-2024（前后对比） ──────────────────────
DIAG_START = "2022-01-01"
DIAG_END   = "2024-12-31"
TOP_N = 30


def get_factor_scores_for_month(close_panel, codes, month_end, industry_map):
    """返回每个因子的原始截面 pd.Series（未合并）"""
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

    # 公共股票集
    common = None
    for fs in norm.values():
        common = set(fs.index) if common is None else common & set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        common = set()

    return norm, common


def compute_ic_for_factor(factor_scores: pd.Series, fwd_ret: pd.Series) -> float:
    """Spearman IC：因子得分 vs 未来收益率"""
    common = factor_scores.index.intersection(fwd_ret.index)
    if len(common) < 10:
        return np.nan
    ic, _ = spearmanr(factor_scores[common], fwd_ret[common])
    return ic


def main():
    # ── 加载数据 ──────────────────────────────────────────────
    members_df = pd.read_parquet(MEMBERS_FILE)
    all_codes  = members_df["con_code"].unique().tolist()

    print(f"加载收盘价面板...")
    close_panel = load_close_panel(codes=all_codes)
    print(f"面板：{close_panel.shape}  {close_panel.index[0].date()} ~ {close_panel.index[-1].date()}")

    print("预加载财务数据...")
    for i, code in enumerate(all_codes, 1):
        get_fina(code)
        if i % 200 == 0:
            print(f"  {i}/{len(all_codes)}")

    print("加载行业映射...")
    industry_map = get_industry_map()
    print(f"  {len(industry_map)} 只\n")

    # ── 逐月诊断 ──────────────────────────────────────────────
    close_sub = close_panel.loc[DIAG_START:DIAG_END]
    nat_ends  = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = []
    for i, month_end in enumerate(monthly_last[:-1]):
        next_end  = monthly_last[i + 1]
        month_end = pd.Timestamp(month_end)
        next_end  = pd.Timestamp(next_end)

        pit_members = load_members_pit(month_end, members_file=MEMBERS_FILE)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        norm, common = get_factor_scores_for_month(close_panel, available, month_end, industry_map)
        n_common = len(common)

        # 各因子覆盖率
        factor_coverage = {fname: len(fs) for fname, fs in norm.items()}

        # 如果公共集够，计算 Top30
        if n_common >= TOP_N and common:
            total_w = sum(ICIR_WEIGHTS.get(f, 0) for f in norm)
            score = pd.Series(0.0, index=list(common))
            for fname, fs in norm.items():
                w = ICIR_WEIGHTS.get(fname, 0) / total_w
                score += fs[list(common)] * w

            selected = score.nlargest(TOP_N).index.tolist()
        else:
            selected = []

        # 未来收益（用于 IC 计算）
        fwd_ret = {}
        for code in available:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[code] = p1 / p0 - 1
        fwd_ret = pd.Series(fwd_ret)

        # 各因子本月 IC
        ic_per_factor = {}
        for fname, fs in norm.items():
            ic_per_factor[fname] = compute_ic_for_factor(fs, fwd_ret)

        # 基准收益（等权 available，只取有价格的股票）
        bm_codes = [c for c in available if c in fwd_ret.index]
        bm_ret = fwd_ret[bm_codes].mean() if bm_codes else np.nan

        # 策略收益（Top30 等权）
        if selected:
            strat_ret = fwd_ret[selected].mean() if fwd_ret[selected].notna().any() else np.nan
        else:
            strat_ret = np.nan

        excess = strat_ret - bm_ret if pd.notna(strat_ret) and pd.notna(bm_ret) else np.nan

        # 行业分布（Top30）
        top30_sectors = [industry_map.get(c, "未知") for c in selected]
        sector_counts = pd.Series(top30_sectors).value_counts() if top30_sectors else pd.Series()
        top_sector     = sector_counts.index[0] if len(sector_counts) > 0 else "N/A"
        top_sector_pct = sector_counts.iloc[0] / TOP_N if len(sector_counts) > 0 else 0.0

        rec = {
            "date":            month_end,
            "year":            month_end.year,
            "n_available":     len(available),
            "n_common":        n_common,
            "n_selected":      len(selected),
            "strat_ret":       strat_ret,
            "bm_ret":          bm_ret,
            "excess":          excess,
            "top_sector":      top_sector,
            "top_sector_pct":  top_sector_pct,
            **{f"cov_{k}": v for k, v in factor_coverage.items()},
            **{f"ic_{k}": v for k, v in ic_per_factor.items()},
        }
        records.append(rec)

    df = pd.DataFrame(records).set_index("date")

    # ── 输出 ──────────────────────────────────────────────────
    print("=" * 65)
    print("  假设1：公共股票集大小（5因子均有数据的股票数）")
    print("=" * 65)
    print(f"\n  {'年份':<6} {'月均available':>14} {'月均公共集':>12} {'覆盖率':>8}")
    for y in sorted(df["year"].unique()):
        yr = df[df["year"] == y]
        print(f"  {y:<6} {yr['n_available'].mean():>14.0f} {yr['n_common'].mean():>12.0f} "
              f"{(yr['n_common'] / yr['n_available']).mean():>8.1%}")

    print(f"\n  各因子月均覆盖（公共集相关）:")
    cov_cols = [c for c in df.columns if c.startswith("cov_")]
    print(f"  {'年份':<6}", end="")
    for c in cov_cols:
        print(f"  {c[4:]:>18}", end="")
    print()
    for y in sorted(df["year"].unique()):
        yr = df[df["year"] == y]
        print(f"  {y:<6}", end="")
        for c in cov_cols:
            print(f"  {yr[c].mean():>18.0f}", end="")
        print()

    print("\n" + "=" * 65)
    print("  假设2：Top30 行业集中度")
    print("=" * 65)
    print(f"\n  {'年份':<6} {'月均top行业占比':>16} {'最常出现top行业'}")
    for y in sorted(df["year"].unique()):
        yr = df[df["year"] == y]
        most_common = yr["top_sector"].mode()[0] if len(yr) > 0 else "N/A"
        print(f"  {y:<6} {yr['top_sector_pct'].mean():>16.1%}  {most_common}")

    # 2023 月度行业 Top
    print(f"\n  2023年每月 Top30 中占比最高的行业：")
    yr2023 = df[df["year"] == 2023]
    for dt, row in yr2023.iterrows():
        print(f"    {dt.strftime('%Y-%m')}  excess={row['excess']*100:+.2f}%  "
              f"top行业={row['top_sector']}({row['top_sector_pct']:.1%})  n_common={row['n_common']:.0f}")

    print("\n" + "=" * 65)
    print("  假设3：各因子月均 IC（2022-2024 年度分拆）")
    print("=" * 65)
    ic_cols = [c for c in df.columns if c.startswith("ic_")]
    print(f"\n  {'年份':<6}", end="")
    for c in ic_cols:
        print(f"  {c[3:]:>18}", end="")
    print(f"  {'excess/月':>10}")
    for y in sorted(df["year"].unique()):
        yr = df[df["year"] == y]
        print(f"  {y:<6}", end="")
        for c in ic_cols:
            print(f"  {yr[c].mean():>+18.3f}", end="")
        print(f"  {yr['excess'].mean()*100:>+9.2f}%")

    print("\n" + "=" * 65)
    print("  月度超额流水（2023）")
    print("=" * 65)
    for dt, row in yr2023.iterrows():
        print(f"  {dt.strftime('%Y-%m')}  strat={row['strat_ret']*100:+.2f}%  "
              f"bm={row['bm_ret']*100:+.2f}%  excess={row['excess']*100:+.2f}%")


if __name__ == "__main__":
    main()
