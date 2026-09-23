"""
第四十六轮候选①②③④因子截面IC验证（异常换手率atr / 残差动量 / 残差反转 / 零成交天数）

来源：子agent深挖Hou/Qiao/Zhang《Finding Anomalies in China》+
Jansen/Swinkels/Zhou(2021)等论文体系产出的4个候选（2026-09-20）。

**候选②补测说明**：一度误判②Residual Momentum与`factor_ic_resmom.py`
（2026-07-03，列为有效因子）重复——但复核发现该脚本docstring写"残差动量"，
实际代码里`compute_momentum`只是简单累积收益，从未实现过对市场beta取残差的
逻辑，docstring描述与代码不一致。真正的残差动量此前从未测试过，本脚本补测。

四个因子定义：
- atr（异常换手率）：当月换手率均值 / 过去12个月换手率均值。换手率代理沿用
  `factor_ic_liquidity.py`的 amount/circ_mv 构造方式（不还原真实流通股数，
  但Rank IC只依赖排序）。区别于该脚本已测的"换手率水平值"，atr测的是
  "相对自身历史基线的异常放大程度"，方向：异常放大 -> 未来收益走低（负向）。
- resmom（残差动量，126/252日+skip21）：对全市场等权均值做OLS取beta后的
  残差累积收益（LOO方式剔除自身，不带截距项），窗口与`factor_ic_resmom.py`
  的skip-month动量对齐，唯一区别是先剔除市场beta。方向：正向（残差动量强
  的股票预期未来涨得更多）。
- resrev（残差反转，短期版）：同样的残差计算逻辑，21日窗口不skip。机制是
  短期反转而非中长期动量，方向：负向（残差累积涨幅大的股票预期未来回落）。
- zerotd（零成交天数占比）：过去N个自然交易日窗口内，该股票未出现在
  stock_daily面板（因停牌等原因缺失）的交易日数 / 窗口理论交易日数。
  方向：零成交占比越高 -> 流动性越差，暂按负向处理（越差流动性预期收益越低）。

**关键实现坑（调试中发现）**：残差累积收益必须用 (1+resid).cumprod()-1，
不能用 sum(resid)——OLS若带截距项，残差和恒等于0（最小二乘正交性质），
用sum会让因子恒为0导致IC全部NaN。本脚本回归时不带截距项，且用累乘法。

验证方式：每月末截面 Spearman Rank IC（因子值 vs 下月收益率），沿用项目既有
factor_ic_*.py方法论（沪深300/中证500成分股，月度截面）。
入选阈值：|IC均值|>=0.03 且年度同向占比>=60%（项目既定阈值）。

用法：
  cd a_stock/backtest
  python factor_ic_v46_candidates.py               # 默认跑全部
  python factor_ic_v46_candidates.py --index hs300
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

STOCK_DIR = DATA_DIR / "stock_daily"
VALUATION_FILE = DATA_DIR / "valuation_monthly.parquet"

OUTPUT_DIR = pathlib.Path(__file__).parent / "results" / "factor_ic_v46_candidates"

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

# atr：当月窗口21日 / 基准窗口252日（约12个月）
ATR_SHORT_WINDOW = 21
ATR_LONG_WINDOW = 252

# resrev：残差反转窗口（短期，不skip）
RESREV_WINDOW = 21

# resmom：残差动量窗口（中长期，skip21规避短期反转），与factor_ic_resmom.py
# 的skip-month动量窗口配置对齐，唯一区别是这里用残差累积收益而非简单累积收益
RESMOM_CONFIGS = {
    "resmom_126s21": {"window": 126, "skip": 21},
    "resmom_252s21": {"window": 252, "skip": 21},
}

# zerotd：零成交天数统计窗口（约6个月自然交易日）
ZEROTD_WINDOW = 126

FACTOR_CONFIG = {
    "atr":           {"name": "异常换手率（21日/252日换手率均值比）",  "direction": -1},
    "resrev":        {"name": "残差反转（21日残差累积收益）",           "direction": -1},
    "resmom_126s21": {"name": "残差动量（126日skip21残差累积收益）",    "direction": +1},
    "resmom_252s21": {"name": "残差动量（252日skip21残差累积收益）",    "direction": +1},
    "zerotd":        {"name": "零成交天数占比（126日窗口）",            "direction": -1},
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

def load_field_panel(codes: list[str], field: str) -> pd.DataFrame:
    """读取指定字段的宽格式面板（index=trade_date，columns=ts_code），字段取自 stock_daily"""
    frames = {}
    for code in codes:
        path = STOCK_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", field])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames[code] = df.set_index("trade_date")[field]
    if not frames:
        raise FileNotFoundError(f"没有找到任何 {field} 数据")
    return pd.DataFrame(frames).sort_index()


def load_circ_mv_panel() -> pd.DataFrame:
    """流通市值月度快照，宽格式（index=trade_date月末近似日，columns=ts_code）"""
    df = pd.read_parquet(VALUATION_FILE)[["trade_date", "ts_code", "circ_mv"]].dropna()
    return df.pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index()


def load_presence_panel(codes: list[str], full_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    构造"是否有交易记录"的0/1面板（宽格式，index=full_calendar，columns=ts_code）。
    stock_daily没有单独的停牌标记字段，个股缺失某交易日的记录即代表当日停牌/未成交。
    """
    presence = {}
    for code in codes:
        path = STOCK_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date"])
            trade_dates = pd.to_datetime(df["trade_date"])
            presence[code] = pd.Series(1, index=trade_dates)
    if not presence:
        raise FileNotFoundError("没有找到任何 stock_daily 数据")
    panel = pd.DataFrame(presence)
    # reindex到全市场交易日历，缺失记录的交易日填0（=当日停牌/零成交）
    panel = panel.reindex(full_calendar).fillna(0).astype(int)
    return panel


# ── 因子计算（日频面板，滚动窗口后在月末截面取值） ─────────

def compute_turnover_daily(amount_panel: pd.DataFrame, circ_mv_panel: pd.DataFrame) -> pd.DataFrame:
    """换手率代理 = amount / circ_mv（circ_mv 月度快照按日前向填充对齐到日频）"""
    circ_mv_daily = circ_mv_panel.reindex(amount_panel.index, method="ffill")
    common_cols = amount_panel.columns.intersection(circ_mv_daily.columns)
    return amount_panel[common_cols] / circ_mv_daily[common_cols].replace(0, np.nan)


def compute_atr_at(turnover_daily: pd.DataFrame, month_end: pd.Timestamp) -> pd.Series:
    """
    异常换手率atr = 当月（短窗口）换手率均值 / 过去12个月（长窗口）换手率均值。
    长窗口包含短窗口，是"相对自身历史基线"的动态比值，区别于换手率水平值本身。
    """
    hist = turnover_daily.loc[:month_end]
    if len(hist) < ATR_LONG_WINDOW + 2:
        return pd.Series(dtype=float)
    short_mean = hist.iloc[-ATR_SHORT_WINDOW:].mean()
    long_mean = hist.iloc[-ATR_LONG_WINDOW:].mean()
    return (short_mean / long_mean.replace(0, np.nan)).dropna()


def compute_residual_cum_return(
    close_panel: pd.DataFrame, month_end: pd.Timestamp, window: int, skip: int = 0
) -> pd.Series:
    """
    共享逻辑：对全市场等权均值做OLS取beta后的残差累积收益（LOO方式剔除自身）。
    beta项：ret_i = beta_i * market_loo_i + resid_i（不带截距项——如果加截距项，
    OLS残差和恒等于0，会把"跑赢/跑输市场的累积幅度"这部分信息完全抹掉，
    只留下与beta估计误差同量级的噪声，是本轮调试中发现的关键坑）。
    残差累积收益用 (1+resid).cumprod()-1，不能用sum(resid)。
    skip>0 时跳过最近skip个交易日（规避短期反转，与`factor_ic_resmom.py`
    的skip-month动量语义一致）。
    """
    hist = close_panel.loc[:month_end]
    if len(hist) < window + skip + 2:
        return pd.Series(dtype=float)

    if skip > 0:
        hist = hist.iloc[:-skip]
    window_close = hist.iloc[-(window + 1):]
    ret = window_close.pct_change().dropna(how="all")
    if ret.empty:
        return pd.Series(dtype=float)

    n = ret.shape[1]
    market_sum = ret.sum(axis=1)
    # LOO：每只股票对应的"市场"是剔除自身后的等权均值
    loo_market = (market_sum.values[:, None] - ret.fillna(0).values) / max(n - 1, 1)
    loo_market = pd.DataFrame(loo_market, index=ret.index, columns=ret.columns)

    resid_cum = {}
    for col in ret.columns:
        y = ret[col].dropna()
        x = loo_market[col].loc[y.index]
        common = y.index.intersection(x.dropna().index)
        if len(common) < window // 2:
            continue
        y_c, x_c = y.loc[common].values, x.loc[common].values
        # 不带截距项的简单回归：beta = sum(x*y) / sum(x*x)
        denom = np.dot(x_c, x_c)
        if denom < 1e-12:
            continue
        beta = np.dot(x_c, y_c) / denom
        resid = y_c - beta * x_c
        resid_cum[col] = np.prod(1 + resid) - 1

    return pd.Series(resid_cum).dropna()


def compute_resrev_at(close_panel: pd.DataFrame, month_end: pd.Timestamp) -> pd.Series:
    """
    残差反转（短期）：21日窗口残差累积收益，窗口短、不skip最近月，
    与残差动量（126/252日+skip21）的中长期动量机制区分开。
    """
    return compute_residual_cum_return(close_panel, month_end, RESREV_WINDOW)


def compute_zerotd_at(presence_panel: pd.DataFrame, month_end: pd.Timestamp) -> pd.Series:
    """零成交天数占比 = 过去126个自然交易日窗口内缺失记录的天数 / 窗口长度"""
    hist = presence_panel.loc[:month_end]
    if len(hist) < ZEROTD_WINDOW + 2:
        return pd.Series(dtype=float)
    window = hist.iloc[-ZEROTD_WINDOW:]
    zero_ratio = 1 - window.mean()  # mean=1表示每天都有记录，1-mean=零成交占比
    return zero_ratio.dropna()


# ── 月度 IC 计算 ──────────────────────────────────────────

def compute_monthly_ic(
    close_panel: pd.DataFrame,
    turnover_daily: pd.DataFrame,
    presence_panel: pd.DataFrame,
    members_file: pathlib.Path,
) -> dict[str, pd.DataFrame]:
    close_panel = close_panel.loc[START_DATE:END_DATE]

    nat_month_ends = close_panel.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_panel.index[close_panel.index <= m][-1]
        for m in nat_month_ends
        if len(close_panel.index[close_panel.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    records = {fkey: [] for fkey in FACTOR_CONFIG}

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

        factor_values = {
            "atr": compute_atr_at(turnover_daily[[c for c in available if c in turnover_daily.columns]], month_end),
            "resrev": compute_resrev_at(close_panel[available], month_end),
            "zerotd": compute_zerotd_at(presence_panel[[c for c in available if c in presence_panel.columns]], month_end),
        }
        for rkey, rcfg in RESMOM_CONFIGS.items():
            factor_values[rkey] = compute_residual_cum_return(
                close_panel[available], month_end, window=rcfg["window"], skip=rcfg["skip"]
            )

        for fkey, factor in factor_values.items():
            if factor.empty:
                continue
            cfg = FACTOR_CONFIG[fkey]
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
    年度同向占比：按项目既定口径，是"各年度IC均值与全样本IC均值符号一致"的
    年份占比，不是月度符号占比。
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


def plot_ic_results(ic_results: dict, output_dir: pathlib.Path, title_prefix: str = "") -> None:
    n = len(ic_results)
    if n == 0:
        return
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"IC 图已保存：{out_path}")


# ── 主流程 ────────────────────────────────────────────────

def run_one_index(
    index_key: str,
    close_panel: pd.DataFrame,
    turnover_daily: pd.DataFrame,
    presence_panel: pd.DataFrame,
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
    ic_results = compute_monthly_ic(close_panel, turnover_daily, presence_panel, members_file)

    summary_rows = []
    for fkey, ic_df in ic_results.items():
        if ic_df.empty:
            print(f"    {FACTOR_CONFIG[fkey]['name']}：无有效数据")
            continue
        stats = summarize_ic(ic_df["ic"], fkey)
        summary_rows.append(stats)
        print(f"    {stats['因子']:<35} IC均值={stats['IC均值']:+.4f}  ICIR={stats['ICIR']:+.3f}  "
              f"IC>0={stats['IC>0占比']}  年度同向占比={stats['年度同向占比']}  "
              f"n={stats['样本月数']}月  {'通过初筛' if stats['通过初筛'] else '未达阈值'}")
        print_annual_ic(ic_df["ic"], stats["因子"])
        ic_df.to_csv(out_dir / f"ic_{fkey}.csv")

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows).set_index("因子")
    summary_df.to_csv(out_dir / "ic_summary.csv")

    valid_results = {k: v for k, v in ic_results.items() if not v.empty}
    if valid_results:
        plot_ic_results(valid_results, out_dir, title_prefix=index_name)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="第四十六轮候选①②③④因子截面IC验证")
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
    amount_panel = load_field_panel(all_codes, "amount")
    print(f"面板大小：{close_panel.shape}  "
          f"（{close_panel.index[0].date()} ~ {close_panel.index[-1].date()}）")

    print("加载流通市值月度快照...")
    circ_mv_panel = load_circ_mv_panel()

    print("计算换手率日频面板...")
    turnover_daily = compute_turnover_daily(amount_panel, circ_mv_panel)

    print("构造停牌/成交记录面板（零成交天数用）...")
    full_calendar = close_panel.index
    presence_panel = load_presence_panel(all_codes, full_calendar)

    all_summaries = {}
    for key in index_keys:
        name = INDEX_CONFIG[key]["name"]
        print(f"\n{'='*60}")
        print(f"指数：{name}（{key}）")
        print(f"{'='*60}")
        summary = run_one_index(key, close_panel, turnover_daily, presence_panel)
        if not summary.empty:
            all_summaries[name] = summary

    if all_summaries:
        print(f"\n{'='*70}")
        print("全量汇总（第四十六轮候选①②③④ Rank IC，月度截面，2016-2026）")
        print(f"{'='*70}")
        for name, df in all_summaries.items():
            print(f"\n--- {name} ---")
            print(df.to_string())
        print()
        print("判定标准：|IC均值|>=0.03 且年度同向占比>=60% 为通过初筛")

    print(f"\n输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
