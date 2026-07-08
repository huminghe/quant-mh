"""
盈利质量因子截面 IC 验证（第二轮）
- OCF/NI：经营现金流 / 净利润（fina_indicator.ocf_to_profit）
- 应计盈余（Accruals）：(△流动资产 - △现金 - △流动负债 + △短期借款) / 总资产
- 盈利增速稳定性：过去 8 个季度 netprofit_yoy 的标准差（取负，低波动=稳定）
- 行业内 EP 排名：在申万行业内对 EP 截面排名后计算 IC（消除行业配置暴露）

数据来源：
  - fina_indicator：ocf_to_profit、netprofit_yoy（已在 financials/ 目录，需扩展字段）
  - balancesheet：total_assets、total_cur_assets、money_cap、total_cur_liab、st_borr
  - stock_basic：industry（申万行业，用于行业内排名）

用法：
  cd a_stock/backtest
  python factor_ic_quality_v2.py               # 默认跑 hs500
  python factor_ic_quality_v2.py --index hs300
  python factor_ic_quality_v2.py --factor ocf  # 只跑某个因子
"""

import sys
import argparse
import pathlib
import warnings
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tushare as ts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR
from fetch_financials import load_financials, FINANCIALS_DIR

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_quality_v2"

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"

MIN_STOCKS_PER_CROSS = 50
MIN_STOCKS_IN_SECTOR = 5     # 行业内排名最少股票数

BS_DIR = DATA_DIR / "balancesheet"   # 资产负债表缓存目录

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
    "ocf": {
        "name": "OCF/NI（现金流质量）",
        "direction": 1,
        "description": "经营现金流 / 净利润（越高越好，正向因子）",
    },
    "accruals": {
        "name": "应计盈余（Accruals，取负）",
        "direction": -1,
        "description": "应计盈余 = (△流动资产-△现金-△流动负债+△短期借款)/总资产，取负表示应计越低越好",
    },
    "profit_stability": {
        "name": "盈利增速稳定性（取负）",
        "direction": -1,
        "description": "过去 8 季度净利润YoY增速的标准差取负（稳定=好）",
    },
    "ep_sector": {
        "name": "行业内 EP 排名",
        "direction": 1,
        "description": "EP 在申万行业内截面排名（消除行业配置暴露）",
    },
}


# ── tushare 初始化 ─────────────────────────────────────────

_pro = None

def get_pro():
    global _pro
    if _pro is None:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro


# ── 资产负债表缓存 ─────────────────────────────────────────

_bs_cache: dict[str, pd.DataFrame] = {}

def load_balancesheet(ts_code: str) -> pd.DataFrame:
    """加载资产负债表（带本地缓存）"""
    if ts_code in _bs_cache:
        return _bs_cache[ts_code]

    path = BS_DIR / f"{ts_code}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["ann_date"] = pd.to_datetime(df["ann_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        _bs_cache[ts_code] = df
        return df

    return pd.DataFrame()


def fetch_and_cache_balancesheet(codes: list[str]) -> None:
    """批量拉取并缓存资产负债表"""
    BS_DIR.mkdir(parents=True, exist_ok=True)
    pro = get_pro()
    fields = ("ts_code,ann_date,end_date,total_assets,total_cur_assets,"
              "money_cap,total_cur_liab,st_borr")

    missing = [c for c in codes if not (BS_DIR / f"{c}.parquet").exists()]
    if not missing:
        print(f"  资产负债表缓存已就绪（{len(codes)} 只）")
        return

    print(f"  下载资产负债表：{len(missing)} 只（共 {len(codes)} 只）...")
    ok = 0
    for i, code in enumerate(missing, 1):
        try:
            df = pro.balancesheet(ts_code=code, start_date="20130101", fields=fields)
            if df is not None and not df.empty:
                df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
                df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
                df = (df.sort_values("ann_date")
                        .drop_duplicates(subset=["ts_code", "end_date"], keep="last")
                        .sort_values(["end_date", "ann_date"])
                        .reset_index(drop=True))
                df.to_parquet(BS_DIR / f"{code}.parquet", index=False)
                ok += 1
        except Exception:
            pass
        time.sleep(0.35)
        if i % 100 == 0:
            print(f"    资产负债表进度：{i}/{len(missing)}")
    print(f"  资产负债表下载完成：{ok}/{len(missing)}")


# ── 行业映射（stock_basic.industry）─────────────────────────

_industry_map: dict[str, str] | None = None

def get_industry_map() -> dict[str, str]:
    global _industry_map
    if _industry_map is not None:
        return _industry_map

    cache_path = DATA_DIR / "stock_industry.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        _industry_map = df.set_index("ts_code")["industry"].to_dict()
        return _industry_map

    print("  下载行业映射（stock_basic）...")
    pro = get_pro()
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,industry")
    df_d = pro.stock_basic(exchange="", list_status="D",
                           fields="ts_code,industry")
    df_p = pro.stock_basic(exchange="", list_status="P",
                           fields="ts_code,industry")
    all_df = pd.concat([df, df_d, df_p], ignore_index=True)
    all_df = all_df.drop_duplicates("ts_code")
    all_df.to_parquet(cache_path, index=False)
    _industry_map = all_df.set_index("ts_code")["industry"].to_dict()
    print(f"  行业映射已缓存：{len(_industry_map)} 只")
    return _industry_map


# ── 财务数据工具 ───────────────────────────────────────────

_fina_cache: dict[str, pd.DataFrame] = {}

def get_fina(ts_code: str) -> pd.DataFrame:
    if ts_code not in _fina_cache:
        _fina_cache[ts_code] = load_financials(ts_code)
    return _fina_cache[ts_code]


def get_fina_pit(ts_code: str, as_of: pd.Timestamp, field: str):
    df = get_fina(ts_code)
    if df.empty or field not in df.columns:
        return None
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    if valid.empty:
        return None
    return float(valid.iloc[-1][field])


def get_fina_history(ts_code: str, as_of: pd.Timestamp,
                     field: str, n: int = 8) -> pd.Series:
    """取截至 as_of 最近 n 条有效财务记录"""
    df = get_fina(ts_code)
    if df.empty or field not in df.columns:
        return pd.Series(dtype=float)
    valid = df[(df["ann_date"] <= as_of) & df[field].notna()]
    return valid[field].iloc[-n:].reset_index(drop=True)


# ── 应计盈余计算（基于资产负债表）──────────────────────────

def compute_accruals_pit(ts_code: str, as_of: pd.Timestamp) -> float | None:
    """
    应计盈余 = (△流动资产 - △现金 - △流动负债 + △短期借款) / 总资产
    使用相邻两个报告期的资产负债表差值，点对时 ann_date <= as_of
    """
    df = load_balancesheet(ts_code)
    if df.empty:
        return None

    valid = df[df["ann_date"] <= as_of].dropna(
        subset=["total_assets", "total_cur_assets", "money_cap", "total_cur_liab"]
    )
    if len(valid) < 2:
        return None

    curr = valid.iloc[-1]
    prev = valid.iloc[-2]

    total_assets = curr["total_assets"]
    if pd.isna(total_assets) or total_assets <= 0:
        return None

    delta_cur_assets = curr["total_cur_assets"] - prev["total_cur_assets"]
    delta_cash       = curr["money_cap"] - prev["money_cap"]
    delta_cur_liab   = curr["total_cur_liab"] - prev["total_cur_liab"]

    # st_borr 可能为 NaN（无短期借款）
    curr_st = curr["st_borr"] if pd.notna(curr.get("st_borr")) else 0.0
    prev_st = prev["st_borr"] if pd.notna(prev.get("st_borr")) else 0.0
    delta_st_borr = curr_st - prev_st

    accruals = (delta_cur_assets - delta_cash - delta_cur_liab + delta_st_borr) / total_assets
    if np.isnan(accruals) or np.isinf(accruals):
        return None
    return float(accruals)


# ── 截面因子计算 ───────────────────────────────────────────

def compute_ocf_cross(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        v = get_fina_pit(code, month_end, "ocf_to_profit")
        if v is not None and not np.isnan(v):
            values[code] = v
    return pd.Series(values)


def compute_accruals_cross(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    values = {}
    for code in codes:
        v = compute_accruals_pit(code, month_end)
        if v is not None:
            values[code] = v
    return pd.Series(values)


def compute_profit_stability_cross(codes: list[str], month_end: pd.Timestamp) -> pd.Series:
    """盈利增速标准差（取负方向：稳定=好）"""
    values = {}
    for code in codes:
        hist = get_fina_history(code, month_end, "netprofit_yoy", n=8)
        if len(hist) < 4:
            continue
        std = hist.std()
        if pd.notna(std) and std >= 0:
            values[code] = std  # direction=-1 会取负
    return pd.Series(values)


def compute_ep_sector_cross(codes: list[str], month_end: pd.Timestamp,
                            close_row: pd.Series,
                            industry_map: dict) -> pd.Series:
    """
    行业内 EP 排名（0-1分位数），消除行业配置暴露。
    先计算每只股票的 EP，再在行业内做 percentile rank。
    """
    ep_raw = {}
    for code in codes:
        eps = get_fina_pit(code, month_end, "eps")
        if eps is None or np.isnan(eps):
            continue
        price = close_row.get(code)
        if price is None or np.isnan(price) or price <= 0:
            continue
        ep_raw[code] = eps / price

    if not ep_raw:
        return pd.Series(dtype=float)

    ep_series = pd.Series(ep_raw)
    sector_series = pd.Series({c: industry_map.get(c, "未知") for c in ep_series.index})

    # 行业内 percentile rank
    ranks = {}
    for sector, group in ep_series.groupby(sector_series):
        if sector == "未知" or len(group) < MIN_STOCKS_IN_SECTOR:
            # 行业内股票太少，用全截面 rank 兜底
            for c in group.index:
                ranks[c] = None
        else:
            # 0-1 标准化排名
            pct_rank = group.rank(pct=True)
            for c, v in pct_rank.items():
                ranks[c] = v

    # 过滤掉兜底的 None
    return pd.Series({c: v for c, v in ranks.items() if v is not None})


# ── 通用工具 ──────────────────────────────────────────────

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
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return ic


# ── 月度 IC 计算（通用）───────────────────────────────────

def compute_monthly_ic_for_factor(
    close_panel: pd.DataFrame,
    factor_key: str,
    members_file: pathlib.Path,
    industry_map: dict | None = None,
) -> pd.DataFrame:
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    cfg = FACTOR_CONFIG[factor_key]
    records = []

    for i, month_end in enumerate(monthly_last[:-1]):
        next_end = monthly_last[i + 1]

        pit_members = load_members_pit(month_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        close_row = close_panel[available].loc[month_end].dropna()
        avail_with_price = list(close_row.index)

        # 计算截面因子
        if factor_key == "ocf":
            raw = compute_ocf_cross(avail_with_price, month_end)
        elif factor_key == "accruals":
            raw = compute_accruals_cross(avail_with_price, month_end)
        elif factor_key == "profit_stability":
            raw = compute_profit_stability_cross(avail_with_price, month_end)
        elif factor_key == "ep_sector":
            raw = compute_ep_sector_cross(avail_with_price, month_end,
                                          close_row, industry_map or {})
        else:
            continue

        if len(raw) < MIN_STOCKS_PER_CROSS:
            continue

        # 方向调整
        factor = raw * cfg["direction"]
        factor = winsorize(factor)
        factor = standardize(factor)

        # 下月收益
        fwd_prices = close_panel[available].loc[next_end].dropna()
        common = close_row.index.intersection(fwd_prices.index).intersection(factor.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        fwd_ret = fwd_prices[common] / close_row[common] - 1
        ic = cross_section_rank_ic(factor[common], fwd_ret)
        records.append({
            "date":         month_end,
            "ic":           ic,
            "n_stocks":     len(common),
        })

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 统计汇总 ──────────────────────────────────────────────

def summarize_ic(ic_series: pd.Series, factor_key: str) -> dict:
    cfg = FACTOR_CONFIG[factor_key]
    clean = ic_series.dropna()
    icir = clean.mean() / clean.std() if clean.std() > 0 else np.nan
    return {
        "因子":       cfg["name"],
        "样本月数":   len(clean),
        "IC均值":     round(clean.mean(), 4),
        "IC标准差":   round(clean.std(), 4),
        "ICIR":       round(icir, 3),
        "IC>0占比":   f"{(clean > 0).mean() * 100:.1f}%",
        "|IC|>0.02":  f"{(clean.abs() > 0.02).mean() * 100:.1f}%",
        "说明":       cfg["description"],
    }


# ── 画图 ──────────────────────────────────────────────────

def plot_ic_results(ic_results: dict, output_dir: pathlib.Path,
                    title_prefix: str = "") -> None:
    n = len(ic_results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (fkey, df) in enumerate(ic_results.items()):
        ax_bar, ax_cum = axes[row_idx]
        ic = df["ic"].dropna()
        fname = FACTOR_CONFIG[fkey]["name"]

        for ax in [ax_bar, ax_cum]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            for label in [ax.yaxis.label, ax.xaxis.label, ax.title]:
                label.set_color("white")

        colors = ["#ef4444" if v < 0 else "#22c55e" for v in ic]
        ax_bar.bar(ic.index, ic.values, color=colors, width=20, alpha=0.8)
        ax_bar.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax_bar.axhline(ic.mean(), color="#facc15", linewidth=1.5,
                       label=f"均值 {ic.mean():.4f}")
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
    out_dir = OUTPUT_DIR / index_key
    out_dir.mkdir(parents=True, exist_ok=True)

    industry_map = get_industry_map() if "ep_sector" in factor_keys else {}

    ic_results   = {}
    summary_rows = []

    for fkey in factor_keys:
        fname = FACTOR_CONFIG[fkey]["name"]
        print(f"  [{cfg['name']}] {fname}...")
        ic_df = compute_monthly_ic_for_factor(
            close_panel, fkey, cfg["members_file"], industry_map
        )
        if ic_df.empty:
            print("    无有效数据")
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
        plot_ic_results(ic_results, out_dir, title_prefix=cfg["name"])
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="盈利质量因子截面IC验证（第二轮）")
    parser.add_argument("--index", choices=["hs300", "hs500", "all"], default="hs500")
    parser.add_argument("--factor",
                        choices=list(FACTOR_CONFIG.keys()) + ["all"],
                        default="all")
    parser.add_argument("--no-fetch-bs", action="store_true",
                        help="跳过资产负债表下载（已有缓存时使用）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_keys  = list(INDEX_CONFIG.keys()) if args.index == "all" else [args.index]
    factor_keys = list(FACTOR_CONFIG.keys()) if args.factor == "all" else [args.factor]

    # 收集股票代码
    all_codes: set[str] = set()
    for key in index_keys:
        mf = INDEX_CONFIG[key]["members_file"]
        if mf.exists():
            m = pd.read_parquet(mf)
            all_codes.update(m["con_code"].unique())

    print(f"加载收盘价面板（共 {len(all_codes)} 只股票）...")
    close_panel = load_close_panel(codes=list(all_codes))
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）\n")

    # 预加载财务缓存（fina_indicator，已有）
    print("预加载 fina_indicator 缓存...")
    for i, code in enumerate(all_codes, 1):
        get_fina(code)
        if i % 200 == 0:
            print(f"  财务缓存：{i}/{len(all_codes)}")
    print()

    # 如果需要应计盈余，预先下载资产负债表
    if "accruals" in factor_keys and not args.no_fetch_bs:
        fetch_and_cache_balancesheet(list(all_codes))
        # 预加载 bs 缓存
        print("  预加载资产负债表缓存...")
        for code in all_codes:
            load_balancesheet(code)
        print()

    for key in index_keys:
        print(f"{'='*60}")
        print(f"指数：{INDEX_CONFIG[key]['name']}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, factor_keys)
        if not summary.empty:
            print(f"\n--- {INDEX_CONFIG[key]['name']} 汇总 ---")
            print(summary.to_string())

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
