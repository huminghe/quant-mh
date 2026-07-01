"""
宏观因子 IC 验证脚本（2026-07）

验证：PMI、M2同比、社融当月新增 是否对 A股行业ETF 次月收益有预测力
方法：月度 Rank IC（斯皮尔曼相关系数），单因子 + 宏观综合打分

输出：
  - 各因子月度 IC 序列、IC均值、ICIR、IC>0占比
  - 综合宏观打分 IC
  - IC 时序图

注意：
  - 宏观数据是截面共同信号（所有ETF看同一个PMI），不是横截面因子
  - 所以这里的"IC"是时序IC：当月宏观打分 vs 所有ETF次月等权平均收益
  - 同时也计算横截面维度：宏观强月 vs 弱月，ETF平均收益差异
"""

import sys
import time
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))

from fetch_data import load_close_matrix, init_pro
from etf_universe import ETF_CODES

# ── 参数 ──────────────────────────────────────────────────
START_DATE    = "2016-01-01"
MACRO_START   = "201601"   # tushare month 格式
MACRO_END     = "202606"
RESULTS_DIR   = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 数据获取 ──────────────────────────────────────────────

def fetch_pmi(pro) -> pd.Series:
    """制造业综合 PMI，返回月末时间戳索引的 Series"""
    df = pro.cn_pmi(start_m=MACRO_START, end_m=MACRO_END)
    df["date"] = pd.to_datetime(df["MONTH"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").set_index("date")
    return df["PMI010000"].astype(float).rename("pmi")


def fetch_m2(pro) -> pd.Series:
    """M2 同比增速"""
    df = pro.cn_m(start_m=MACRO_START, end_m=MACRO_END)
    df["date"] = pd.to_datetime(df["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").set_index("date")
    return df["m2_yoy"].astype(float).rename("m2_yoy")


def fetch_sf(pro) -> pd.Series:
    """社融当月新增（亿元）"""
    df = pro.sf_month(start_m=MACRO_START, end_m=MACRO_END)
    df["date"] = pd.to_datetime(df["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").set_index("date")
    return df["inc_month"].astype(float).rename("sf_inc")


# ── 宏观综合打分（参考湘财证券方法） ──────────────────────

def build_macro_score(pmi: pd.Series, m2: pd.Series, sf: pd.Series) -> pd.DataFrame:
    """
    构建综合宏观打分（0-10分制）：
      PMI 分：(PMI - 49) * 5，截断到 [0, 10]
      M2 分：M2 同比相对过去 24 个月历史分位数，映射到 [0, 10]
      社融分：当月新增相对过去 24 个月历史分位数，映射到 [0, 10]
      综合分 = 三者等权平均
    """
    common_idx = pmi.index.intersection(m2.index).intersection(sf.index)

    # PMI 打分
    pmi_score = ((pmi.reindex(common_idx) - 49) * 5).clip(0, 10)

    # M2 分位数打分（滚动24月）
    m2_s = m2.reindex(common_idx)
    m2_score = m2_s.rolling(24, min_periods=12).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() * 10, raw=False
    )

    # 社融分位数打分（滚动24月）
    sf_s = sf.reindex(common_idx)
    sf_score = sf_s.rolling(24, min_periods=12).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() * 10, raw=False
    )

    macro_score = (pmi_score + m2_score + sf_score) / 3

    return pd.DataFrame({
        "pmi":         pmi.reindex(common_idx),
        "m2_yoy":      m2_s,
        "sf_inc":      sf_s,
        "pmi_score":   pmi_score,
        "m2_score":    m2_score,
        "sf_score":    sf_score,
        "macro_score": macro_score,
    })


# ── ETF 月度收益矩阵 ───────────────────────────────────────

def build_monthly_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    构建 ETF 月度收益矩阵，index = 月末日期，columns = ETF代码
    月末取最后一个交易日收盘价
    """
    monthly = close.resample("ME").last()
    return monthly.pct_change().dropna(how="all")


# ── 时序 IC 计算 ──────────────────────────────────────────

def calc_time_series_ic(macro_df: pd.DataFrame, etf_ret: pd.DataFrame) -> pd.DataFrame:
    """
    时序 IC：每个月，宏观打分 vs 所有行业ETF 次月等权平均收益
    返回：每月宏观打分和 ETF 平均收益
    """
    # 只取行业ETF（排除宽基），宏观因子对行业轮换的指引更清晰
    from etf_universe import SECTOR_ETFS
    sector_codes = [c for c in SECTOR_ETFS.keys() if c in etf_ret.columns]

    # 次月收益：把 etf_ret 往前移一期（t 期的宏观 → t+1 期收益）
    fwd_ret = etf_ret[sector_codes].mean(axis=1).shift(-1)  # 所有行业ETF等权平均

    rows = []
    for date in macro_df.index:
        if date not in fwd_ret.index or pd.isna(fwd_ret[date]):
            continue
        macro_val = macro_df.loc[date, "macro_score"]
        if pd.isna(macro_val):
            continue
        rows.append({
            "date":        date,
            "macro_score": macro_val,
            "pmi_score":   macro_df.loc[date, "pmi_score"],
            "m2_score":    macro_df.loc[date, "m2_score"],
            "sf_score":    macro_df.loc[date, "sf_score"],
            "fwd_ret":     fwd_ret[date],
        })

    return pd.DataFrame(rows).set_index("date")


def calc_ic_stats(ts_df: pd.DataFrame, factor_col: str) -> dict:
    """计算单个因子的时序 IC 统计量"""
    valid = ts_df[[factor_col, "fwd_ret"]].dropna()
    if len(valid) < 10:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_pos_ratio": np.nan, "n": len(valid)}

    ic_series = []
    # 时序IC：对所有样本计算斯皮尔曼相关（整体一个数）
    # 但更常用的是：滚动12个月的相关系数序列，查看稳定性
    # 这里直接报全样本相关和滚动序列
    from scipy.stats import spearmanr
    rho, p = spearmanr(valid[factor_col], valid["fwd_ret"])

    # 逐月 IC：每个月作为一个"样本点"，计算因子值与次月收益的相关
    # 时序只有一列，所以我们用滚动12个月计算局部相关
    n = len(valid)
    rolling_ic = []
    for i in range(12, n):
        window = valid.iloc[i-12:i]
        r, _ = spearmanr(window[factor_col], window["fwd_ret"])
        rolling_ic.append((valid.index[i], r))

    ic_s = pd.Series([v for _, v in rolling_ic], index=[d for d, _ in rolling_ic])

    return {
        "ic_mean":      rho,
        "ic_p":         p,
        "icir":         ic_s.mean() / ic_s.std() if ic_s.std() > 0 else 0,
        "ic_pos_ratio": (ic_s > 0).mean(),
        "n":            n,
        "rolling_ic":   ic_s,
    }


def calc_quintile_return(ts_df: pd.DataFrame, factor_col: str, n_groups: int = 3) -> pd.DataFrame:
    """
    把宏观打分按分位数分成 n_groups 组，统计各组次月 ETF 平均收益
    宏观是整体信号，用三分组（弱/中/强）比较直观
    """
    valid = ts_df[[factor_col, "fwd_ret"]].dropna()
    valid = valid.copy()
    valid["group"] = pd.qcut(valid[factor_col], q=n_groups,
                              labels=["弱", "中", "强"][:n_groups])
    return valid.groupby("group", observed=True)["fwd_ret"].agg(
        均值=lambda x: f"{x.mean()*100:.2f}%",
        中位数=lambda x: f"{x.median()*100:.2f}%",
        胜率=lambda x: f"{(x>0).mean():.1%}",
        样本数="count"
    )


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("宏观因子 IC 验证（PMI + M2 + 社融）")
    print("=" * 60)

    # 初始化 tushare
    pro = init_pro()

    # 获取宏观数据
    print("\n获取宏观数据...")
    pmi = fetch_pmi(pro);  time.sleep(0.5)
    m2  = fetch_m2(pro);   time.sleep(0.5)
    sf  = fetch_sf(pro)
    print(f"  PMI: {len(pmi)} 个月，{pmi.index[0].date()} ~ {pmi.index[-1].date()}")
    print(f"  M2:  {len(m2)} 个月")
    print(f"  社融: {len(sf)} 个月")

    # 构建宏观打分
    macro_df = build_macro_score(pmi, m2, sf)
    macro_df = macro_df[macro_df.index >= START_DATE]
    print(f"\n宏观打分样本：{len(macro_df)} 个月")
    print(macro_df[["pmi", "pmi_score", "m2_yoy", "m2_score", "sf_inc", "sf_score", "macro_score"]].tail(6).to_string())

    # 加载 ETF 收益矩阵
    print("\n加载 ETF 数据...")
    close = load_close_matrix()
    close = close[close.index >= START_DATE]
    etf_ret = build_monthly_returns(close)
    print(f"ETF 收益矩阵：{etf_ret.shape}，{etf_ret.index[0].date()} ~ {etf_ret.index[-1].date()}")

    # 构建时序对齐表
    ts_df = calc_time_series_ic(macro_df, etf_ret)
    print(f"\n对齐后样本：{len(ts_df)} 个月")

    # 计算各因子 IC
    print("\n" + "=" * 60)
    print("IC 统计结果（时序相关：宏观打分 vs 行业ETF次月等权平均收益）")
    print("=" * 60)

    factors = {
        "PMI 打分":    "pmi_score",
        "M2 打分":     "m2_score",
        "社融 打分":   "sf_score",
        "综合宏观打分": "macro_score",
    }

    results = {}
    for label, col in factors.items():
        stats = calc_ic_stats(ts_df, col)
        results[label] = stats
        print(f"\n{label}:")
        print(f"  全样本IC (Spearman): {stats['ic_mean']:.4f}  p={stats['ic_p']:.3f}")
        print(f"  滚动12月 ICIR:       {stats['icir']:.3f}")
        print(f"  IC>0 占比:           {stats['ic_pos_ratio']:.1%}  (N={stats['n']})")

    # 分组收益对比
    print("\n" + "=" * 60)
    print("宏观状态分组 vs ETF 次月等权收益（三分组：弱/中/强）")
    print("=" * 60)
    for label, col in factors.items():
        print(f"\n{label}:")
        print(calc_quintile_return(ts_df, col).to_string())

    # 绘图
    print("\n生成图表...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("宏观因子 IC 验证（PMI + M2 + 社融）", fontsize=14, fontweight="bold")

    # 左列：宏观指标历史走势
    ax = axes[0, 0]
    ts_df["pmi_score"].plot(ax=ax, color="steelblue")
    ax.axhline(5, color="red", linestyle="--", alpha=0.5, label="中性线(5)")
    ax.set_title("PMI 打分（0-10）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ts_df["m2_score"].plot(ax=ax, color="green")
    ts_df["sf_score"].plot(ax=ax, color="orange", alpha=0.7)
    ax.axhline(5, color="red", linestyle="--", alpha=0.5)
    ax.set_title("M2打分 vs 社融打分")
    ax.legend(["M2打分", "社融打分"], fontsize=8)

    ax = axes[2, 0]
    ts_df["macro_score"].plot(ax=ax, color="purple")
    ax.axhline(5, color="red", linestyle="--", alpha=0.5, label="中性线(5)")
    ax.set_title("综合宏观打分")
    ax.legend(fontsize=8)

    # 右列：滚动IC序列
    for i, (label, col) in enumerate(factors.items()):
        if i >= 3:
            break
        ax = axes[i, 1]
        rolling_ic = results[label].get("rolling_ic", pd.Series())
        if not rolling_ic.empty:
            rolling_ic.plot(ax=ax, color="darkblue", alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(rolling_ic.mean(), color="red", linestyle="--",
                       label=f"均值={rolling_ic.mean():.3f}")
            ax.fill_between(rolling_ic.index, rolling_ic, 0,
                            where=rolling_ic > 0, alpha=0.3, color="green")
            ax.fill_between(rolling_ic.index, rolling_ic, 0,
                            where=rolling_ic <= 0, alpha=0.3, color="red")
        ax.set_title(f"{label} 滚动12月IC")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "macro_ic_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{out_path}")

    # 输出综合结论
    print("\n" + "=" * 60)
    print("结论判断标准（时序IC）：")
    print("  |IC| > 0.1 且 IC>0占比 > 55%  → 有效，可接入策略")
    print("  |IC| 在 0.05-0.1               → 弱有效，谨慎使用")
    print("  |IC| < 0.05 或 IC>0占比 ≈ 50%  → 无效，放弃")
    print("=" * 60)

    macro_stats = results["综合宏观打分"]
    ic = macro_stats["ic_mean"]
    pos = macro_stats["ic_pos_ratio"]
    if abs(ic) > 0.1 and pos > 0.55:
        verdict = "有效 — 建议接入策略验证"
    elif abs(ic) > 0.05:
        verdict = "弱有效 — 谨慎，建议先做参数敏感性"
    else:
        verdict = "无效 — 放弃，不做全量回测"
    print(f"\n综合宏观打分：IC={ic:.4f}，IC>0占比={pos:.1%}")
    print(f"→ 结论：{verdict}")


if __name__ == "__main__":
    main()
