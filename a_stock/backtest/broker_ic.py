"""
分析师推荐景气 IC 验证脚本（2026-07）

数据来源：tushare broker_recommend（月度各券商推荐股票汇总，2021年起有效）
信号逻辑：将每月被推荐股票数量聚合到行业ETF层面，
         统计每只ETF对应行业的「推荐覆盖度」及其月度变化，
         作为分析师景气预期的代理变量

方法：月度截面 Rank IC（斯皮尔曼相关系数）
  - 因子值：ETF对应行业当月推荐股票数量 / 行业成分股总数（覆盖率）
  - 因子变化：当月覆盖率 - 3个月前覆盖率（边际变化，预期修正代理）
  - 预测目标：该ETF次月收益

注意：
  - broker_recommend 只有推荐股票列表，没有评级方向（买入/中性/卖出），
    默认全部是正向推荐（卖方研报惯例）
  - 2021年前无数据，样本仅约48个月，结论需谨慎
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))

from fetch_data import load_close_matrix, init_pro
from etf_universe import ETF_UNIVERSE, SECTOR_ETFS

# ── 参数 ──────────────────────────────────────────────────
START_MONTH  = "202101"    # broker_recommend 数据起始
END_MONTH    = "202605"    # 最后一个有完整次月收益的月份
RESULTS_DIR  = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── ETF → 申万一级行业 映射 ────────────────────────────────
# 手工维护：每只行业ETF对应的申万一级行业代码
# ⚠️ 已废弃/不再维护（2026-07-19标注）：本字典未被任何其他脚本引用，是孤立死代码。
# 与申万行业分层候选池方案做一致性对照时发现，14/28条注释描述的品种已与
# market_etf_meta.parquet当前真实名称对不上（如159611.SZ注释"煤炭ETF"实际是电力公用事业ETF），
# 不要再拿本字典当校验基准，详见 a_stock/docs/research.md"申万一级行业分层抽样候选池"小节。
ETF_TO_SW_INDUSTRY = {
    "515000.SH": "801080.SI",   # 计算机ETF → 计算机
    "512760.SH": "801080.SI",   # 半导体ETF → 电子（用801080近似）
    "159995.SZ": "801080.SI",   # 芯片ETF   → 电子
    "515330.SH": "801200.SI",   # 新能源车ETF → 汽车
    "516160.SH": "801730.SI",   # 新能源ETF  → 电力设备
    "159629.SZ": "801730.SI",   # 光伏ETF    → 电力设备
    "159596.SZ": "801730.SI",   # 风电ETF    → 电力设备
    "512010.SH": "801150.SI",   # 医疗卫生ETF → 医药生物
    "512170.SH": "801150.SI",   # 医疗器械ETF → 医药生物
    "159992.SZ": "801150.SI",   # CXO ETF   → 医药生物
    "512800.SH": "801780.SI",   # 银行ETF    → 银行
    "512880.SH": "801790.SI",   # 证券ETF    → 非银金融
    "159931.SZ": "801180.SI",   # 地产ETF    → 房地产
    "512980.SH": "801760.SI",   # 传媒ETF    → 传媒
    "159869.SZ": "801760.SI",   # 游戏ETF    → 传媒
    "515030.SH": "801730.SI",   # 新能源ETF华夏 → 电力设备
    "159628.SZ": "801890.SI",   # 机器人ETF  → 机械设备
    "516670.SH": "801050.SI",   # 稀土ETF    → 有色金属
    "159975.SZ": "801660.SI",   # 军工ETF    → 国防军工
    "512660.SH": "801660.SI",   # 军工ETF工银 → 国防军工
    "512400.SH": "801050.SI",   # 有色金属ETF → 有色金属
    "159928.SZ": "801120.SI",   # 消费ETF    → 食品饮料
    "515700.SH": "801120.SI",   # 食品饮料ETF → 食品饮料
    "159997.SZ": "801120.SI",   # 白酒ETF    → 食品饮料
    "159801.SZ": "801010.SI",   # 农业ETF    → 农林牧渔
    "515220.SH": "801030.SI",   # 化工ETF    → 基础化工
    "516950.SH": "801080.SI",   # 港股互联网ETF → 计算机(近似)
    "513050.SH": "801080.SI",   # 中概互联ETF → 计算机(近似)
    "159611.SZ": "801040.SI",   # 煤炭ETF    → 煤炭
}

# 申万一级行业代码 → 名称（用于展示）
SW_NAMES = {
    "801010.SI": "农林牧渔",
    "801020.SI": "采掘",
    "801030.SI": "基础化工",
    "801040.SI": "煤炭",
    "801050.SI": "有色金属",
    "801080.SI": "电子/计算机",
    "801110.SI": "家用电器",
    "801120.SI": "食品饮料",
    "801130.SI": "纺织服装",
    "801140.SI": "轻工制造",
    "801150.SI": "医药生物",
    "801160.SI": "公用事业",
    "801170.SI": "交通运输",
    "801180.SI": "房地产",
    "801200.SI": "汽车",
    "801210.SI": "商业贸易",
    "801230.SI": "综合",
    "801710.SI": "建筑材料",
    "801720.SI": "建筑装饰",
    "801730.SI": "电力设备",
    "801740.SI": "机械设备",
    "801750.SI": "国防军工",  # 注：801660 是另一个编码
    "801760.SI": "传媒",
    "801770.SI": "通信",
    "801780.SI": "银行",
    "801790.SI": "非银金融",
    "801890.SI": "机械设备",
    "801660.SI": "国防军工",
}


# ── 数据获取 ──────────────────────────────────────────────

def fetch_all_broker_recommend(pro) -> pd.DataFrame:
    """拉取所有月度分析师推荐数据，返回合并 DataFrame"""
    months = pd.period_range(START_MONTH, END_MONTH, freq="M").strftime("%Y%m").tolist()
    frames = []
    for i, month in enumerate(months):
        try:
            df = pro.broker_recommend(month=month)
            if not df.empty:
                frames.append(df)
            time.sleep(0.3)
            if (i + 1) % 10 == 0:
                print(f"  已拉取 {i+1}/{len(months)} 个月...")
        except Exception as e:
            print(f"  {month} 失败: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_industry_member(pro) -> pd.DataFrame:
    """
    获取申万一级行业成分股（用于计算行业成分股总数）
    返回：ts_code → industry_code 的映射表
    """
    industry_codes = list(set(ETF_TO_SW_INDUSTRY.values()))
    frames = []
    for code in industry_codes:
        try:
            df = pro.index_member(index_code=code)
            if not df.empty:
                df["industry_code"] = code
                frames.append(df[["con_code", "industry_code"]])
            time.sleep(0.3)
        except Exception as e:
            print(f"  行业成分股 {code} 失败: {e}")
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates("con_code")


# ── 信号构建 ──────────────────────────────────────────────

def build_industry_coverage(recommend_df: pd.DataFrame,
                             member_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建月度行业推荐覆盖率：
      coverage = 当月被推荐的行业内股票数 / 行业总成分股数
    返回：index=月末日期，columns=行业代码
    """
    # 建立股票 → 行业映射
    stk_to_ind = dict(zip(member_df["con_code"], member_df["industry_code"]))
    # 行业总成分股数
    ind_total = member_df.groupby("industry_code").size()

    # 逐月统计
    months = recommend_df["month"].unique()
    rows = []
    for month in sorted(months):
        df_m = recommend_df[recommend_df["month"] == month].copy()
        # 给每只推荐股票打行业标签
        df_m["industry"] = df_m["ts_code"].map(stk_to_ind)
        df_m = df_m.dropna(subset=["industry"])
        # 计算覆盖率
        rec_count = df_m.groupby("industry").size()
        coverage = (rec_count / ind_total).fillna(0)

        date = pd.to_datetime(month, format="%Y%m") + pd.offsets.MonthEnd(0)
        row = {"date": date}
        row.update(coverage.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def coverage_to_etf_factor(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    将行业覆盖率映射到 ETF，并计算边际变化（3个月差分）
    返回：含 coverage 和 coverage_chg 的 DataFrame
    """
    results = {}
    for etf_code, ind_code in ETF_TO_SW_INDUSTRY.items():
        if ind_code not in coverage_df.columns:
            continue
        cov = coverage_df[ind_code].rename(f"{etf_code}_cov")
        chg = cov - cov.shift(3)
        chg.name = f"{etf_code}_chg"
        results[etf_code] = pd.DataFrame({"coverage": cov, "coverage_chg": chg})

    # 合并成两个矩阵
    cov_matrix = pd.DataFrame({k: v["coverage"]  for k, v in results.items()})
    chg_matrix = pd.DataFrame({k: v["coverage_chg"] for k, v in results.items()})
    return cov_matrix, chg_matrix


# ── IC 计算 ───────────────────────────────────────────────

def calc_cross_section_ic(factor_matrix: pd.DataFrame,
                           etf_ret: pd.DataFrame) -> pd.Series:
    """
    截面 Rank IC：每个月，对所有ETF计算 factor 排名 vs 次月收益 排名 的斯皮尔曼相关
    返回月度 IC 序列
    """
    from scipy.stats import spearmanr

    # 次月收益
    fwd_ret = etf_ret.shift(-1)

    ic_series = {}
    common_dates = factor_matrix.index.intersection(fwd_ret.index)

    for date in common_dates:
        f = factor_matrix.loc[date].dropna()
        r = fwd_ret.loc[date].dropna()
        common_etfs = f.index.intersection(r.index)
        if len(common_etfs) < 5:
            continue
        rho, _ = spearmanr(f[common_etfs], r[common_etfs])
        if not np.isnan(rho):
            ic_series[date] = rho

    return pd.Series(ic_series).sort_index()


def print_ic_stats(ic: pd.Series, label: str):
    """打印 IC 统计摘要"""
    if ic.empty or len(ic) < 3:
        print(f"{label}: 样本不足，无法计算")
        return
    print(f"\n{label}（截面 Rank IC）:")
    print(f"  IC 均值:    {ic.mean():.4f}")
    print(f"  IC 标准差:  {ic.std():.4f}")
    print(f"  ICIR:       {ic.mean()/ic.std():.3f}" if ic.std() > 0 else "  ICIR: N/A")
    print(f"  IC>0 占比:  {(ic>0).mean():.1%}")
    print(f"  样本月数:   {len(ic)}")


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("分析师推荐景气 IC 验证（broker_recommend 聚合）")
    print("=" * 60)

    pro = init_pro()

    # 获取推荐数据
    print("\n拉取 broker_recommend 月度数据...")
    recommend_df = fetch_all_broker_recommend(pro)
    if recommend_df.empty:
        print("无数据，退出")
        return
    print(f"共获取推荐记录：{len(recommend_df)} 条，"
          f"{recommend_df['month'].min()} ~ {recommend_df['month'].max()}")

    # 获取行业成分股
    print("\n获取申万行业成分股...")
    member_df = fetch_industry_member(pro)
    print(f"成分股映射：{len(member_df)} 条")
    if member_df.empty:
        print("成分股数据为空，退出")
        return

    # 构建行业推荐覆盖率
    print("\n构建行业推荐覆盖率...")
    coverage_df = build_industry_coverage(recommend_df, member_df)
    print(f"行业覆盖率矩阵：{coverage_df.shape}")

    # 映射到 ETF
    cov_matrix, chg_matrix = coverage_to_etf_factor(coverage_df)
    print(f"ETF覆盖率矩阵：{cov_matrix.shape}，边际变化矩阵：{chg_matrix.shape}")

    # 加载 ETF 月度收益
    print("\n加载 ETF 收益数据...")
    close = load_close_matrix()
    monthly = close.resample("ME").last()
    etf_ret = monthly.pct_change()
    # 只取行业ETF（同时确保在 cov_matrix 中存在）
    sector_codes = [c for c in ETF_TO_SW_INDUSTRY.keys()
                    if c in etf_ret.columns and c in cov_matrix.columns]
    etf_ret_sector = etf_ret[sector_codes]
    print(f"行业ETF月度收益矩阵：{etf_ret_sector.shape}")
    print(f"有效行业ETF（同时有覆盖率和收益数据）：{len(sector_codes)} 只")

    # 计算 IC
    print("\n计算截面 Rank IC...")
    ic_coverage = calc_cross_section_ic(cov_matrix[sector_codes], etf_ret_sector)
    ic_chg      = calc_cross_section_ic(chg_matrix[sector_codes], etf_ret_sector)

    print_ic_stats(ic_coverage, "推荐覆盖率（绝对水平）")
    print_ic_stats(ic_chg,      "推荐覆盖率变化（3M差分，预期修正代理）")

    # 赋值供后续绘图用（统一变量名）
    etf_ret = etf_ret_sector

    # 绘图
    print("\n生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("分析师推荐景气 IC 验证", fontsize=13, fontweight="bold")

    # IC 时序
    for ax, ic, title in [
        (axes[0, 0], ic_coverage, "推荐覆盖率 月度截面 IC"),
        (axes[0, 1], ic_chg,      "推荐覆盖率变化 月度截面 IC"),
    ]:
        if not ic.empty:
            ic.plot(ax=ax, color="steelblue", alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(ic.mean(), color="red", linestyle="--",
                       label=f"均值={ic.mean():.3f}")
            ax.fill_between(ic.index, ic, 0,
                            where=ic > 0, alpha=0.3, color="green")
            ax.fill_between(ic.index, ic, 0,
                            where=ic <= 0, alpha=0.3, color="red")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # 各行业覆盖率热力图（近12个月）
    ax = axes[1, 0]
    recent_cov = cov_matrix.tail(12)
    if not recent_cov.empty:
        # 取有数据的列，显示前15个
        show_cols = recent_cov.columns[:15]
        im = ax.imshow(recent_cov[show_cols].T.values,
                       aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(show_cols)))
        ax.set_yticklabels([c.split(".")[0] for c in show_cols], fontsize=6)
        ax.set_xticks(range(len(recent_cov)))
        ax.set_xticklabels(
            [d.strftime("%y%m") for d in recent_cov.index], rotation=45, fontsize=6
        )
        ax.set_title("近12月各ETF推荐覆盖率热力图")
        plt.colorbar(im, ax=ax)

    # IC 累积
    ax = axes[1, 1]
    if not ic_chg.empty:
        ic_chg.cumsum().plot(ax=ax, color="purple", label="覆盖率变化 累积IC")
    if not ic_coverage.empty:
        ic_coverage.cumsum().plot(ax=ax, color="orange", label="覆盖率水平 累积IC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("IC 累积走势")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "broker_ic_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{out_path}")

    # 结论
    print("\n" + "=" * 60)
    print("结论判断标准（截面 Rank IC）：")
    print("  |IC均值| > 0.05 且 IC>0占比 > 55%  → 有效")
    print("  |IC均值| < 0.03 或 IC>0占比 ≈ 50%  → 无效")
    print("  注意：样本仅约48个月，统计显著性偏弱")
    print("=" * 60)

    for ic, label in [(ic_coverage, "推荐覆盖率"), (ic_chg, "覆盖率变化（预期修正）")]:
        if ic.empty:
            continue
        ic_mean = ic.mean()
        pos = (ic > 0).mean()
        if abs(ic_mean) > 0.05 and pos > 0.55:
            verdict = "有效"
        elif abs(ic_mean) > 0.03:
            verdict = "弱有效，谨慎"
        else:
            verdict = "无效"
        print(f"\n{label}: IC均值={ic_mean:.4f}，IC>0占比={pos:.1%} → {verdict}")


if __name__ == "__main__":
    main()
