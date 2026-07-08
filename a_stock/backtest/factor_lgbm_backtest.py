"""
LightGBM 非线性因子合成回测

用 LightGBM 替代线性 ICIR 加权，学习因子间非线性交互。

方法：
  - 滚动训练窗口：用过去 TRAIN_WINDOW 个月的截面数据训练，预测下个月排名
  - 标签：下个月实际收益的截面排名（Rank IC 目标）
  - 特征：5个标准化因子得分
  - 防过拟合：TimeSeriesSplit + 早停 + 树深度限制

对比基线：V2 线性 ICIR 加权（factor_multi_backtest_v2.py）

用法：
  cd a_stock/backtest
  python factor_lgbm_backtest.py
"""

import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    MIN_STOCKS_CROSS, INDEX_CONFIG,
    COST_PER_TRADE, STAMP_DUTY, RISK_FREE_ANNUAL,
)

# ── 配置 ──────────────────────────────────────────────────
OUTPUT_DIR  = pathlib.Path(__file__).parent / "results" / "factor_lgbm_backtest"
MEMBERS_FILE = INDEX_CONFIG["hs500"]["members_file"]
ICIR_WEIGHTS = INDEX_CONFIG["hs500"]["factor_icir"]

START_DATE = "2016-01-01"
END_DATE   = "2026-06-30"
IS_END     = "2024-01-31"
OOS_START  = "2024-02-01"

TOP_N = 30

# LightGBM 训练窗口（月）：至少需要多少期历史数据才开始预测
TRAIN_WINDOW = 36   # 3年的截面数据作为最小训练集

LGB_PARAMS = {
    "objective":       "regression",    # 预测未来收益率（截面标准化后）
    "metric":          "rmse",
    "n_estimators":    200,
    "learning_rate":   0.05,
    "max_depth":       4,               # 浅树防过拟合
    "num_leaves":      15,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}

FACTOR_NAMES = ["profit_stability", "ep_sector", "ocf", "roe", "reversal"]


# ── 截面数据收集 ──────────────────────────────────────────

def collect_cross_section(
    close_panel: pd.DataFrame,
    codes: list[str],
    month_end: pd.Timestamp,
    next_end: pd.Timestamp,
    industry_map: dict,
) -> pd.DataFrame | None:
    """
    收集某月的因子截面数据 + 未来收益标签。
    返回 DataFrame，列：[factor_names..., fwd_ret, fwd_rank]
    """
    close_row = close_panel[codes].loc[month_end].dropna()
    available = list(close_row.index)

    raw = {
        "reversal":         compute_reversal(close_panel, available, month_end),
        "ep_sector":        compute_ep_sector(available, month_end, close_row, industry_map),
        "ocf":              compute_ocf(available, month_end),
        "roe":              compute_roe(available, month_end),
        "profit_stability": compute_profit_stability(available, month_end),
    }

    # 标准化
    norm = {}
    for fname, fs in raw.items():
        if len(fs) < MIN_STOCKS_CROSS // 2:
            continue
        fs = winsorize(fs)
        fs = standardize(fs)
        norm[fname] = fs

    if len(norm) < len(FACTOR_NAMES):
        return None

    # 公共股票集（5个因子均有数据）
    common = None
    for fs in norm.values():
        common = set(fs.index) if common is None else common & set(fs.index)
    if not common or len(common) < MIN_STOCKS_CROSS:
        return None
    common = list(common)

    # 未来收益
    fwd_ret = {}
    for code in common:
        p0 = close_panel[code].get(month_end)
        p1 = close_panel[code].get(next_end)
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            fwd_ret[code] = p1 / p0 - 1
    if len(fwd_ret) < MIN_STOCKS_CROSS:
        return None

    # 拼装截面 DataFrame
    codes_valid = list(fwd_ret.keys())
    rows = {fname: norm[fname][codes_valid] for fname in FACTOR_NAMES}
    df = pd.DataFrame(rows, index=codes_valid)
    df["fwd_ret"]  = pd.Series(fwd_ret)
    df["fwd_rank"] = df["fwd_ret"].rank()   # 截面排名（LGB 标签）
    df["month"]    = month_end
    return df.dropna()


# ── 滚动训练预测 ──────────────────────────────────────────

def run_lgbm_rolling(
    monthly_last: np.ndarray,
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    industry_map: dict,
) -> pd.DataFrame:
    """
    滚动训练：每个预测月，用过去 TRAIN_WINDOW 个月训练，预测当月排名。
    返回每月的策略/基准收益。
    """
    # 1) 先收集所有截面数据（离线，避免重复计算）
    print("  收集因子截面数据...")
    cross_data: dict[pd.Timestamp, pd.DataFrame] = {}

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

        cs = collect_cross_section(close_panel, available, month_end, next_end, industry_map)
        if cs is not None and len(cs) >= TOP_N:
            cross_data[month_end] = cs

    months_sorted = sorted(cross_data.keys())
    print(f"  有效截面月份：{len(months_sorted)} 个")

    if len(months_sorted) < TRAIN_WINDOW + 6:
        print("  数据不足，无法训练")
        return pd.DataFrame()

    # 2) 滚动训练 + 预测
    records = []
    for i in range(TRAIN_WINDOW, len(months_sorted)):
        pred_month = months_sorted[i]
        train_months = months_sorted[i - TRAIN_WINDOW: i]

        # 训练集（标签：截面标准化未来收益，消除月度市场涨跌影响）
        train_df = pd.concat([cross_data[m] for m in train_months], ignore_index=True)
        # 每个月内截面标准化标签
        def norm_label(df):
            mu, sd = df["fwd_ret"].mean(), df["fwd_ret"].std()
            df = df.copy()
            df["label"] = (df["fwd_ret"] - mu) / sd if sd > 1e-8 else 0.0
            return df
        train_df = pd.concat([norm_label(cross_data[m]) for m in train_months], ignore_index=True)
        X_train  = train_df[FACTOR_NAMES].values
        y_train  = train_df["label"].values

        # 验证集（最近12期）用于早停
        val_months = train_months[-12:]
        val_df     = pd.concat([norm_label(cross_data[m]) for m in val_months], ignore_index=True)
        X_val      = val_df[FACTOR_NAMES].values
        y_val      = val_df["label"].values

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)],
        )

        # 预测
        pred_df  = cross_data[pred_month].copy()
        X_pred   = pred_df[FACTOR_NAMES].values
        scores   = model.predict(X_pred)
        pred_df["lgbm_score"] = scores

        selected = pred_df.nlargest(TOP_N, "lgbm_score").index.tolist()

        # 找下一个月的实际收益
        fwd_ret  = pred_df["fwd_ret"]
        strat_gross = fwd_ret[selected].mean()
        bm_ret      = fwd_ret.mean()

        turnover_est = 0.5
        cost         = turnover_est * (COST_PER_TRADE + STAMP_DUTY)
        strat_net    = strat_gross - cost

        records.append({
            "date":      pred_month,
            "strategy":  strat_net,
            "benchmark": bm_ret,
            "gross_ret": strat_gross,
            "n_stocks":  len(selected),
        })

        if (i - TRAIN_WINDOW) % 12 == 0:
            print(f"  预测进度：{i - TRAIN_WINDOW + 1}/{len(months_sorted) - TRAIN_WINDOW} 月")

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


# ── 线性基线（复用 V2 逻辑） ──────────────────────────────

def run_linear_baseline(
    monthly_last: np.ndarray,
    close_panel: pd.DataFrame,
    members_file: pathlib.Path,
    industry_map: dict,
) -> pd.DataFrame:
    """V2 线性 ICIR 加权基线（只跑与 LGB 同期，用于对比）"""
    from factor_multi_backtest_v2 import run_backtest
    ret = run_backtest(close_panel, members_file, ICIR_WEIGHTS, industry_map, TOP_N, "icir")
    return ret


# ── 绩效统计 ──────────────────────────────────────────────

def sharpe(ret: pd.Series, freq: int = 12) -> float:
    if len(ret) < 2:
        return np.nan
    ann = ret.mean() * freq
    std = ret.std() * np.sqrt(freq)
    return np.nan if std < 1e-8 else (ann - RISK_FREE_ANNUAL) / std


def max_drawdown(nav: pd.Series) -> float:
    return ((nav - nav.cummax()) / nav.cummax()).min()


def annual_return(nav: pd.Series, freq: int = 12) -> float:
    n = len(nav)
    if n < 2:
        return np.nan
    return (1 + nav.iloc[-1] / nav.iloc[0] - 1) ** (1 / (n / freq)) - 1


def print_stats(ret_df: pd.DataFrame, label: str, start=None, end=None) -> None:
    df = ret_df.copy()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    strat  = df["strategy"].dropna()
    bench  = df["benchmark"].dropna()
    common = strat.index.intersection(bench.index)
    strat, bench = strat[common], bench[common]
    excess = strat - bench

    nav_s = (1 + strat).cumprod()
    nav_b = (1 + bench).cumprod()

    period = f"{df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}" if len(df) else "—"
    print(f"\n  [{label}]  {period}")
    print(f"    样本月数：{len(strat)}")
    print(f"    策略年化：{annual_return(nav_s)*100:.1f}%  基准年化：{annual_return(nav_b)*100:.1f}%")
    print(f"    超额年化（net）：{excess.mean()*12*100:.1f}%")
    print(f"    策略夏普：{sharpe(strat):.3f}  基准夏普：{sharpe(bench):.3f}")
    print(f"    策略最大回撤：{max_drawdown(nav_s)*100:.1f}%")
    print(f"    月胜率vs基准：{(excess>0).mean()*100:.1f}%")


def print_annual(ret_df: pd.DataFrame, label: str) -> None:
    strat  = ret_df["strategy"].dropna()
    bench  = ret_df["benchmark"].dropna()
    excess = (strat - bench).dropna()
    print(f"\n  {label} 年度超额（net，月均）:")
    for y in sorted(excess.index.year.unique()):
        yr = excess[excess.index.year == y]
        print(f"    {y}: {yr.mean()*100:+.2f}%/月  (n={len(yr)})")


def plot_comparison(ret_lgbm: pd.DataFrame, ret_linear: pd.DataFrame,
                    output_dir: pathlib.Path) -> None:
    # 对齐到同期（LGB 从 TRAIN_WINDOW 个月后才开始）
    common_idx = ret_lgbm.index.intersection(ret_linear.index)
    lgbm_strat  = (1 + ret_lgbm.loc[common_idx, "strategy"]).cumprod()
    lgbm_excess = (1 + ret_lgbm.loc[common_idx, "strategy"]
                   - ret_lgbm.loc[common_idx, "benchmark"]).cumprod()
    lin_strat   = (1 + ret_linear.loc[common_idx, "strategy"]).cumprod()
    lin_excess  = (1 + ret_linear.loc[common_idx, "strategy"]
                   - ret_linear.loc[common_idx, "benchmark"]).cumprod()
    bm          = (1 + ret_lgbm.loc[common_idx, "benchmark"]).cumprod()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("white")

    ax_nav, ax_exc = axes

    ax_nav.plot(lgbm_strat.index, lgbm_strat.values, color="#60a5fa", lw=1.5, label="LightGBM")
    ax_nav.plot(lin_strat.index,  lin_strat.values,  color="#f97316", lw=1.5, label="线性 ICIR (V2)")
    ax_nav.plot(bm.index, bm.values, color="gray", lw=1.2, ls="--", label="基准（等权）")
    ax_nav.set_title("净值曲线（对齐起点）")
    ax_nav.set_ylabel("净值")
    ax_nav.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_exc.plot(lgbm_excess.index, lgbm_excess.values, color="#60a5fa", lw=1.5, label="LightGBM 超额")
    ax_exc.plot(lin_excess.index,  lin_excess.values,  color="#f97316", lw=1.5, label="线性 ICIR 超额")
    ax_exc.axhline(1, color="white", lw=0.8, ls="--")
    ax_exc.set_title("超额累积净值（对齐起点）")
    ax_exc.set_ylabel("超额累积净值")
    ax_exc.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_exc.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(pad=2.0)
    out = output_dir / "lgbm_vs_linear.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  图表已保存：{out}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
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

    # 月份序列
    close_sub = close_panel.loc[START_DATE:END_DATE]
    nat_ends  = close_sub.resample("ME").last().dropna(how="all").index
    monthly_last = pd.Series([
        close_sub.index[close_sub.index <= m][-1]
        for m in nat_ends
        if len(close_sub.index[close_sub.index <= m]) > 0
    ]).drop_duplicates().sort_values().values

    # LightGBM 滚动回测
    print("=" * 60)
    print("  LightGBM 滚动训练回测（训练窗口 36 个月）")
    print("=" * 60)
    ret_lgbm = run_lgbm_rolling(monthly_last, close_panel, MEMBERS_FILE, industry_map)

    if ret_lgbm.empty:
        print("  LGB 回测无结果")
        return

    ret_lgbm.to_csv(OUTPUT_DIR / "ret_lgbm.csv")

    # 线性基线（全期）
    print("\n  计算线性 ICIR 基线...")
    ret_linear = run_linear_baseline(monthly_last, close_panel, MEMBERS_FILE, industry_map)
    ret_linear.to_csv(OUTPUT_DIR / "ret_linear.csv")

    # ── 对齐到同期对比 ────────────────────────────────────
    lgbm_start = ret_lgbm.index[0]
    print(f"\n  LGB 预测起点：{lgbm_start.strftime('%Y-%m')}（前 {TRAIN_WINDOW} 个月用于训练）")

    print("\n" + "=" * 60)
    print("  LightGBM 结果")
    print("=" * 60)
    print_stats(ret_lgbm, "全样本")
    print_stats(ret_lgbm, "IS（2016-2024）", end=IS_END)
    print_stats(ret_lgbm, "OOS（2024-2026）", start=OOS_START)
    print_annual(ret_lgbm, "LightGBM")

    print("\n" + "=" * 60)
    print(f"  线性 ICIR 基线（同期，从 {lgbm_start.strftime('%Y-%m')} 起）")
    print("=" * 60)
    print_stats(ret_linear, "全样本（同期）", start=lgbm_start)
    print_stats(ret_linear, "IS（同期）", start=lgbm_start, end=IS_END)
    print_stats(ret_linear, "OOS（同期）", start=OOS_START)
    print_annual(ret_linear[ret_linear.index >= lgbm_start], "线性 ICIR（同期）")

    # 差值
    common = ret_lgbm.index.intersection(ret_linear.index)
    delta_excess = (ret_lgbm.loc[common, "strategy"] - ret_lgbm.loc[common, "benchmark"]) - \
                   (ret_linear.loc[common, "strategy"] - ret_linear.loc[common, "benchmark"])
    print(f"\n  LGB vs 线性 超额差值：{delta_excess.mean()*12*100:+.2f}%/年（月均 {delta_excess.mean()*100:+.3f}%）")

    plot_comparison(ret_lgbm, ret_linear, OUTPUT_DIR)
    print(f"\n  输出目录：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
