"""
新因子 IC 验证：PEG 和成长动量
- PEG：行业内 (Price/EPS*4) / netprofit_yoy，取负（低PEG=高得分）
- 成长动量（Growth Momentum）：当期 netprofit_yoy 截面排名（高成长=高得分）

两者都是"成长"类因子，理论上在 2019/2023 成长行情不反向。

用法：
  cd a_stock/backtest
  python factor_ic_growth.py
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
    winsorize, standardize,
    MIN_STOCKS_CROSS, MIN_STOCKS_SECTOR, INDEX_CONFIG,
)

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_growth"

MEMBERS_FILE = INDEX_CONFIG["hs500"]["members_file"]


# ── 因子计算 ──────────────────────────────────────────────

def compute_growth_momentum(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """当期净利润YoY增速（越高越好，正向因子）"""
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "netprofit_yoy")
        if v is not None and not np.isnan(v):
            # 过滤极端值：增速超过 500% 或 < -200% 的截断（财务重组/一次性影响）
            if -200 <= v <= 500:
                values[code] = v
    return pd.Series(values)


def compute_peg(codes: list[str], month_end: pd.Timestamp,
                close_row: pd.Series, industry_map: dict) -> pd.Series:
    """
    行业内 PEG = (Price / EPS_annual) / netprofit_yoy
    取负：低 PEG = 高得分（便宜的成长股）

    注意：
    - EPS 用 PIT 最新季报 EPS × 4 年化（粗略，季报 EPS 非年化）
    - netprofit_yoy > 0 才计算（负增速 PEG 无意义）
    - 行业内分位排名（消除行业估值差异）
    """
    peg_raw = {}
    for code in codes:
        eps = get_fina_pit(code, month_end, "eps")
        growth = get_fina_pit(code, month_end, "netprofit_yoy")
        if eps is None or growth is None:
            continue
        if np.isnan(eps) or np.isnan(growth):
            continue
        if eps <= 0 or growth <= 0:
            continue  # 负盈利或负增速 PEG 无意义
        price = close_row.get(code)
        if price is None or np.isnan(price) or price <= 0:
            continue

        # EPS 是最近一期季报，×4 粗略年化
        pe = price / (eps * 4)
        if pe <= 0 or pe > 500:  # 过滤极高 PE
            continue
        peg_raw[code] = pe / growth  # PEG（越低越好）

    if not peg_raw:
        return pd.Series(dtype=float)

    peg_series = pd.Series(peg_raw)
    sector_series = pd.Series({c: industry_map.get(c, "未知") for c in peg_series.index})

    # 行业内分位排名，取负（低 PEG = 高排名 = 高得分）
    ranks = {}
    fallback_rank = (-peg_series).rank(pct=True)  # 全截面兜底
    for code in peg_series.index:
        sector = sector_series[code]
        if sector == "未知":
            ranks[code] = fallback_rank[code]
            continue
        group = peg_series[sector_series == sector]
        if len(group) < MIN_STOCKS_SECTOR:
            ranks[code] = fallback_rank[code]
        else:
            # 行业内低 PEG 排名高
            ranks[code] = (-group).rank(pct=True)[code]

    return pd.Series(ranks)


# ── IC 验证主循环 ──────────────────────────────────────────

def run_ic_validation(close_panel: pd.DataFrame,
                      members_file: pathlib.Path,
                      industry_map: dict) -> None:
    close_sub    = close_panel.loc[START_DATE:END_DATE]
    nat_ends     = close_sub.resample("ME").last().dropna(how="all").index
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

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        close_row = close_panel[available].loc[month_end].dropna()
        avail_with_price = list(close_row.index)

        # 计算两个因子
        growth = compute_growth_momentum(avail_with_price, month_end)
        peg    = compute_peg(avail_with_price, month_end, close_row, industry_map)

        # 未来收益
        fwd_ret = {}
        for code in avail_with_price:
            p0 = close_panel[code].get(month_end)
            p1 = close_panel[code].get(next_end)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[code] = p1 / p0 - 1
        fwd_ret = pd.Series(fwd_ret)

        def calc_ic(factor: pd.Series) -> float:
            common = factor.index.intersection(fwd_ret.index)
            if len(common) < 20:
                return np.nan
            ic, _ = spearmanr(factor[common], fwd_ret[common])
            return ic

        rec = {
            "date":         month_end,
            "year":         month_end.year,
            "ic_growth":    calc_ic(growth),
            "ic_peg":       calc_ic(peg),
            "n_growth":     len(growth),
            "n_peg":        len(peg),
        }
        records.append(rec)

    df = pd.DataFrame(records).set_index("date")

    # ── 输出汇总 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  新因子 IC 汇总（中证500，月度截面，2016-2026）")
    print("=" * 60)

    for col, label in [("ic_growth", "成长动量 (netprofit_yoy)"),
                       ("ic_peg",    "行业内PEG（取负）")]:
        vals = df[col].dropna()
        icir = vals.mean() / vals.std() if vals.std() > 1e-8 else np.nan
        pos  = (vals > 0).mean()
        print(f"\n  {label}")
        print(f"    ICIR：{icir:+.3f}   IC均值：{vals.mean():+.4f}   IC>0占比：{pos:.1%}   n={len(vals)}")

    print("\n  年度 IC 分拆（成长动量 | 行业内PEG）：")
    print(f"  {'年份':<6} {'成长动量IC':>12} {'行业内PEG IC':>14} {'n_growth':>10} {'n_peg':>8}")
    for y in sorted(df["year"].unique()):
        yr = df[df["year"] == y]
        g  = yr["ic_growth"].mean()
        p  = yr["ic_peg"].mean()
        ng = yr["n_growth"].mean()
        np_ = yr["n_peg"].mean()
        print(f"  {y:<6} {g:>+12.4f} {p:>+14.4f} {ng:>10.0f} {np_:>8.0f}")

    # 与现有因子对比（参考V2 IC验证结果）
    print("\n  与 V2 因子 ICIR 对比（参考）：")
    print("    盈利增速稳定性：ICIR +0.322")
    print("    行业内EP：      ICIR +0.321")
    print("    OCF/NI：        ICIR +0.219")
    print("    ROE_TTM：       ICIR +0.195")
    print("    反转（63日）：  ICIR +0.123")

    return df


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

    ic_df = run_ic_validation(close_panel, MEMBERS_FILE, industry_map)
    ic_df.to_csv(OUTPUT_DIR / "ic_growth_factors.csv")
    print(f"\n  输出：{OUTPUT_DIR / 'ic_growth_factors.csv'}")


if __name__ == "__main__":
    main()
