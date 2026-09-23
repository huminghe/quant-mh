"""
V2选股组合（沪深300口径） + 残差动量全子集消融（2026-09-20）

背景：`factor_ic_v46_candidates.py`验证残差动量（对市场等权均值做OLS取
beta后的残差累积收益，126日/252日skip21两个窗口）仅在沪深300通过IC初筛
（IC均值+0.0377/+0.0455，年度同向63.6%/70.0%），中证500未通过。现有多因子
组合消融体系（V2/V3/V4）全部是中证500口径，本脚本另建沪深300专用消融，
不与中证500体系混合。

**注意**：`research_index_enhancement.md`"IC验证第一轮"曾把"残差动量"列为
已验证有效因子，但复核`factor_ic_resmom.py`发现该脚本从未真正实现过残差
逻辑（`FACTOR_CONFIG`只有简单动量三档，docstring描述与代码脱节），历史
`results/factor_ic_resmom/hs300/ic_resmom_126s21.csv`文件IC全为NaN，是当
时一次失败实现的遗留产物。真正的残差动量本轮（v46）才首次补测通过。

因子清单（5个V2既有因子+2个残差动量窗口，共7个）：
- reversal / ep_sector / ocf / roe / profit_stability：沪深300口径，
  计算逻辑与`factor_multi_backtest_v2.py`完全一致（直接复用其函数）
- resmom_126s21 / resmom_252s21：残差动量，计算逻辑复用
  `factor_ic_v46_candidates.py`的`compute_residual_cum_return`

两阶段：
1. 截面相关性检验：7个因子两两截面相关性（pooled月度z-score），确认残差
   动量与reversal（同样基于价格）是否高度冗余
2. 全子集消融：127种非空子集等权合成，Top30月度调仓，全样本/IS/OOS三段
   评估（IS到2024-01，OOS从2024-02），按OOS超额排序

用法：
  cd a_stock/backtest
  python factor_multi_backtest_v5_hs300_resmom.py
"""

import sys
import pathlib
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from factor_multi_backtest_v2 import (  # noqa: E402
    get_industry_map,
    compute_reversal, compute_ep_sector, compute_ocf, compute_roe,
    compute_profit_stability, winsorize, standardize,
    sharpe, max_drawdown,
    COST_PER_TRADE, STAMP_DUTY, IS_END, OOS_START,
    MIN_STOCKS_CROSS, MIN_STOCKS_SECTOR, MIN_HISTORY_QTRS,
)
from factor_ic_v46_candidates import compute_residual_cum_return  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_multi_backtest_v5_hs300_resmom"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

TOP_N = 30

INDEX_KEY    = "hs300"
MEMBERS_FILE = DATA_DIR / "hs300_members.parquet"
INDEX_NAME   = "沪深300"

RESMOM_CONFIGS = {
    "resmom_126s21": {"window": 126, "skip": 21},
    "resmom_252s21": {"window": 252, "skip": 21},
}

FACTOR_NAMES = ["reversal", "ep_sector", "ocf", "roe", "profit_stability",
                "resmom_126s21", "resmom_252s21"]


# ── 单月截面因子计算 ────────────────────────────────────────

def compute_month_factors(
    close_panel: pd.DataFrame,
    codes: list[str],
    month_end: pd.Timestamp,
    industry_map: dict,
) -> dict[str, pd.Series]:
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    raw = {
        "reversal":         compute_reversal(close_panel, available, month_end),
        "ep_sector":        compute_ep_sector(available, month_end, close_row, industry_map),
        "ocf":              compute_ocf(available, month_end),
        "roe":              compute_roe(available, month_end),
        "profit_stability": compute_profit_stability(available, month_end),
    }

    for rkey, rcfg in RESMOM_CONFIGS.items():
        raw[rkey] = compute_residual_cum_return(
            close_panel[available], month_end, window=rcfg["window"], skip=rcfg["skip"]
        )

    norm = {}
    for fname, fs in raw.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm[fname] = fs

    return norm


def get_monthly_dates(close_panel: pd.DataFrame) -> np.ndarray:
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values
    return monthly_last


# ── 数据预计算（跑一次，供相关性检验+消融复用） ──────────────

def precompute_monthly_data(
    close_panel: pd.DataFrame,
    industry_map: dict,
) -> list[dict]:
    monthly_last = get_monthly_dates(close_panel)
    entries = []

    for i, month_end in enumerate(monthly_last[:-1]):
        month_end = pd.Timestamp(month_end)
        next_end  = pd.Timestamp(monthly_last[i + 1])

        pit_members = load_members_pit(month_end, members_file=MEMBERS_FILE)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_CROSS:
            continue

        factors = compute_month_factors(close_panel, available, month_end, industry_map)
        if not factors:
            continue

        close_row = close_panel[available].loc[month_end].dropna()
        fwd_row   = close_panel[available].loc[next_end].dropna()
        common_ret = close_row.index.intersection(fwd_row.index)
        if len(common_ret) < MIN_STOCKS_CROSS:
            continue
        fwd_ret = fwd_row[common_ret] / close_row[common_ret] - 1

        entries.append({
            "month_end": month_end,
            "next_end": next_end,
            "factors": factors,
            "fwd_ret": fwd_ret,
        })

    return entries


# ── 阶段1：截面相关性检验 ───────────────────────────────────

def run_correlation_check(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        factors = entry["factors"]
        common = None
        for fs in factors.values():
            common = set(fs.index) if common is None else common & set(fs.index)
        if not common or len(common) < MIN_STOCKS_CROSS // 2:
            continue
        common = list(common)
        row_df = pd.DataFrame({fname: fs[common] for fname, fs in factors.items()})
        rows.append(row_df)

    pooled = pd.concat(rows, axis=0, ignore_index=True)
    corr = pooled.corr()
    return corr


# ── 阶段2：全子集消融 ──────────────────────────────────────

def run_subset_backtest(entries: list[dict], subset: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for entry in entries:
        factors = entry["factors"]
        present = [f for f in subset if f in factors]
        if not present:
            continue

        common = None
        for f in present:
            fs = factors[f]
            common = set(fs.index) if common is None else common & set(fs.index)
        if not common or len(common) < TOP_N:
            continue
        common = list(common)

        score = pd.Series(0.0, index=common)
        for f in present:
            score += factors[f][common] / len(present)

        selected = score.nlargest(TOP_N).index.tolist()
        fwd_ret = entry["fwd_ret"]

        sel_rets = [fwd_ret[c] for c in selected if c in fwd_ret.index]
        if not sel_rets:
            continue
        gross_ret = np.mean(sel_rets)
        cost = 0.5 * (COST_PER_TRADE + STAMP_DUTY)
        strategy_ret = gross_ret - cost

        benchmark_ret = fwd_ret.mean()

        records.append({
            "date": entry["month_end"],
            "strategy": strategy_ret,
            "benchmark": benchmark_ret,
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def evaluate_subset(ret_df: pd.DataFrame) -> dict:
    if ret_df.empty:
        return None
    strat = ret_df["strategy"]
    bench = ret_df["benchmark"]
    excess = strat - bench

    is_mask  = ret_df.index <= IS_END
    oos_mask = ret_df.index >= OOS_START

    nav_s = (1 + strat).cumprod()

    return {
        "n_months": len(ret_df),
        "全样本超额年化": excess.mean() * 12,
        "全样本夏普": sharpe(strat),
        "全样本最大回撤": max_drawdown(nav_s),
        "IS超额年化": excess[is_mask].mean() * 12 if is_mask.sum() > 0 else np.nan,
        "OOS超额年化": excess[oos_mask].mean() * 12 if oos_mask.sum() > 0 else np.nan,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    members = pd.read_parquet(MEMBERS_FILE)
    codes = members["con_code"].unique().tolist()

    print(f"加载收盘价面板（{INDEX_NAME}，共 {len(codes)} 只股票）...")
    close_panel = load_close_panel(codes=codes)
    print(f"面板大小：{close_panel.shape}")

    print("加载申万行业映射...")
    industry_map = get_industry_map()

    print("预计算月度截面因子（7个因子：反转/行业内EP/OCF/ROE/盈利稳定性/残差动量126s21/残差动量252s21）...")
    entries = precompute_monthly_data(close_panel, industry_map)
    print(f"有效月份数：{len(entries)}\n")

    # ── 阶段1：相关性检验 ──
    print("=" * 60)
    print("阶段1：截面相关性检验（pooled月度z-score）")
    print("=" * 60)
    corr = run_correlation_check(entries)
    print(corr.round(3).to_string())
    corr.to_csv(OUTPUT_DIR / "factor_correlation.csv")

    high_corr_pairs = []
    for i, f1 in enumerate(FACTOR_NAMES):
        for f2 in FACTOR_NAMES[i + 1:]:
            if f1 in corr.index and f2 in corr.columns:
                v = corr.loc[f1, f2]
                if abs(v) > 0.5:
                    high_corr_pairs.append((f1, f2, v))
    if high_corr_pairs:
        print("\n高相关性对（|corr|>0.5，冗余候选）：")
        for f1, f2, v in high_corr_pairs:
            print(f"    {f1} vs {f2}: {v:+.3f}")
    else:
        print("\n无高相关性对（|corr|均<=0.5），7个因子两两基本独立。")

    # ── 阶段2：全子集消融 ──
    print(f"\n{'='*60}")
    print("阶段2：全子集消融（127种非空子集，等权合成，Top30）")
    print(f"{'='*60}")

    results = []
    for r in range(1, len(FACTOR_NAMES) + 1):
        for subset in itertools.combinations(FACTOR_NAMES, r):
            ret_df = run_subset_backtest(entries, subset)
            stats = evaluate_subset(ret_df)
            if stats is None:
                continue
            stats["subset"] = "+".join(subset)
            stats["n_factors"] = len(subset)
            results.append(stats)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)

    print(f"\n共评估 {len(result_df)} 种子集。按 OOS超额年化 排序前15：")
    top15_oos = result_df.sort_values("OOS超额年化", ascending=False).head(15)
    print(top15_oos[["subset", "全样本超额年化", "IS超额年化", "OOS超额年化",
                      "全样本夏普", "全样本最大回撤"]].to_string(index=False,
                      formatters={
                          "全样本超额年化": lambda x: f"{x*100:+.1f}%",
                          "IS超额年化": lambda x: f"{x*100:+.1f}%",
                          "OOS超额年化": lambda x: f"{x*100:+.1f}%",
                          "全样本夏普": lambda x: f"{x:.3f}",
                          "全样本最大回撤": lambda x: f"{x*100:.1f}%",
                      }))

    print(f"\nV2基线（reversal+ep_sector+ocf+roe+profit_stability，不含残差动量）：")
    v2_row = result_df[result_df["subset"].apply(
        lambda s: set(s.split("+")) == {"reversal", "ep_sector", "ocf", "roe", "profit_stability"}
    )]
    if not v2_row.empty:
        print(v2_row[["subset", "全样本超额年化", "IS超额年化", "OOS超额年化",
                       "全样本夏普", "全样本最大回撤"]].to_string(index=False,
                       formatters={
                           "全样本超额年化": lambda x: f"{x*100:+.1f}%",
                           "IS超额年化": lambda x: f"{x*100:+.1f}%",
                           "OOS超额年化": lambda x: f"{x*100:+.1f}%",
                           "全样本夏普": lambda x: f"{x:.3f}",
                           "全样本最大回撤": lambda x: f"{x*100:.1f}%",
                       }))
    else:
        print("  未找到该子集（可能因数据不足被跳过）")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
