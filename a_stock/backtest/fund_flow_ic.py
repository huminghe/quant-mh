"""
ETF 资金净流量 IC 验证脚本（2026-07）

数据来源：tushare fund_share（ETF每日份额数据）
信号逻辑：ETF大额净流出后短期反而上涨（非信息性交易造成短期价格压力）
         → 净流出是反向信号（华泰证券2024研究结论）

构建因子：
  1. flow_chg_1m：当月份额净变化量（当月末 - 上月末）/ 上月末份额
  2. flow_chg_3m：过去3个月份额净变化（相对12个月历史分位数，标准化）

预测目标：次月ETF收益

方法：月度截面 Rank IC
  - 注意：若反向信号有效，IC 应为负值（净流出→次月上涨）

注意事项：
  - fund_share 数据部分ETF 2019年前缺失，以实际有效月份为准
  - 宽基ETF有大量国家队买入（非市场信号），建议主要看行业ETF
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
from etf_universe import ETF_CODES, ETF_UNIVERSE, SECTOR_ETFS

# ── 参数 ──────────────────────────────────────────────────
START_DATE   = "2019-01-01"   # fund_share 较早的有效起始
RESULTS_DIR  = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 数据获取 ──────────────────────────────────────────────

def fetch_fund_share_all(pro, codes: list[str],
                          start_date: str = "20190101") -> pd.DataFrame:
    """
    批量获取所有ETF的日度份额数据
    返回：index=trade_date，columns=ts_code，value=fd_share（万份）
    """
    today = pd.Timestamp.today().strftime("%Y%m%d")
    frames = {}
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = pro.fund_share(ts_code=code, start_date=start_date, end_date=today)
            if not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").set_index("trade_date")
                frames[code] = df["fd_share"].astype(float)
            time.sleep(0.2)
            if i % 10 == 0:
                print(f"  已获取 {i}/{total}...")
        except Exception as e:
            print(f"  {code} 失败: {e}")

    if not frames:
        return pd.DataFrame()

    matrix = pd.DataFrame(frames)
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index()


# ── 因子构建 ──────────────────────────────────────────────

def build_flow_factors(share_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建 ETF 月度资金流因子：
      1. flow_1m：当月份额净变化率 = (月末份额 - 上月末份额) / 上月末份额
      2. flow_3m_pct：过去3月累计变化率相对历史12月分位数（标准化，去除趋势增长效应）

    返回 (flow_1m_matrix, flow_3m_pct_matrix)，index=月末日期，columns=ETF代码
    """
    # 取月末份额
    monthly_share = share_matrix.resample("ME").last()

    # 1. 单月变化率
    flow_1m = monthly_share.pct_change()

    # 2. 3月累计变化率
    flow_3m_raw = monthly_share.pct_change(3)

    # 3. 历史分位数标准化（消除ETF规模增长趋势）
    flow_3m_pct = pd.DataFrame(index=flow_3m_raw.index, columns=flow_3m_raw.columns, dtype=float)
    for i in range(12, len(flow_3m_raw)):
        date = flow_3m_raw.index[i]
        for code in flow_3m_raw.columns:
            hist = flow_3m_raw[code].iloc[i-12:i].dropna()
            curr = flow_3m_raw.iloc[i][code]
            if len(hist) >= 6 and not pd.isna(curr):
                flow_3m_pct.loc[date, code] = (hist < curr).mean()

    return flow_1m, flow_3m_pct


# ── IC 计算 ───────────────────────────────────────────────

def calc_cross_section_ic(factor_matrix: pd.DataFrame,
                           etf_ret: pd.DataFrame,
                           codes: list[str]) -> pd.Series:
    """截面 Rank IC：每月 factor 排名 vs 次月收益排名 的斯皮尔曼相关"""
    from scipy.stats import spearmanr

    fwd_ret = etf_ret[codes].shift(-1)
    ic_series = {}
    common_dates = factor_matrix.index.intersection(fwd_ret.index)

    for date in common_dates:
        f = factor_matrix[codes].loc[date].dropna()
        r = fwd_ret.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        rho, _ = spearmanr(f[common], r[common])
        if not np.isnan(rho):
            ic_series[date] = rho

    return pd.Series(ic_series).sort_index()


def print_ic_stats(ic: pd.Series, label: str):
    if ic.empty or len(ic) < 3:
        print(f"{label}: 样本不足")
        return
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0
    print(f"\n{label}:")
    print(f"  IC 均值:    {ic.mean():.4f}")
    print(f"  IC 标准差:  {ic.std():.4f}")
    print(f"  ICIR:       {icir:.3f}")
    print(f"  IC>0 占比:  {(ic>0).mean():.1%}")
    print(f"  样本月数:   {len(ic)}")


# ── 分组收益分析 ──────────────────────────────────────────

def calc_quintile_returns(factor_matrix: pd.DataFrame,
                           etf_ret: pd.DataFrame,
                           codes: list[str],
                           n_groups: int = 3) -> pd.DataFrame:
    """
    按因子值分组，统计各组次月平均收益
    组1=流出最多（若反向有效，组1应有最高收益）
    """
    fwd_ret = etf_ret[codes].shift(-1)
    common_dates = factor_matrix.index.intersection(fwd_ret.index)

    all_rows = []
    for date in common_dates:
        f = factor_matrix[codes].loc[date].dropna()
        r = fwd_ret.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups * 2:
            continue
        df_tmp = pd.DataFrame({"factor": f[common], "ret": r[common]})
        try:
            df_tmp["group"] = pd.qcut(df_tmp["factor"], q=n_groups,
                                       labels=[f"Q{j+1}" for j in range(n_groups)],
                                       duplicates="drop")
        except ValueError:
            continue
        df_tmp["date"] = date
        all_rows.append(df_tmp)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows)
    return combined.groupby("group", observed=True)["ret"].agg(
        均值=lambda x: f"{x.mean()*100:.2f}%",
        中位数=lambda x: f"{x.median()*100:.2f}%",
        胜率=lambda x: f"{(x>0).mean():.1%}",
        样本数="count"
    )


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ETF 资金净流量 IC 验证")
    print("反向假设：净流出 → 次月上涨（非信息性交易价格压力）")
    print("=" * 60)

    pro = init_pro()

    # 获取所有ETF份额数据
    print(f"\n获取 {len(ETF_CODES)} 只ETF份额数据（{START_DATE} 至今）...")
    start_ts = START_DATE.replace("-", "")
    share_matrix = fetch_fund_share_all(pro, ETF_CODES, start_date=start_ts)

    if share_matrix.empty:
        print("份额数据为空，退出")
        return

    # 过滤有效数据（至少有 24 个月的非NaN）
    monthly_share = share_matrix.resample("ME").last()
    valid_codes = [c for c in monthly_share.columns
                   if monthly_share[c].notna().sum() >= 24]
    print(f"有效ETF（≥24个月数据）：{len(valid_codes)} 只")

    # 只取行业ETF（排除宽基，国家队干扰小）
    sector_valid = [c for c in valid_codes if c in SECTOR_ETFS]
    all_valid    = valid_codes
    print(f"  行业ETF：{len(sector_valid)} 只")
    print(f"  全部ETF：{len(all_valid)} 只")

    # 构建因子
    print("\n构建月度资金流因子...")
    flow_1m, flow_3m_pct = build_flow_factors(share_matrix[valid_codes])
    flow_1m    = flow_1m[flow_1m.index    >= START_DATE]
    flow_3m_pct = flow_3m_pct[flow_3m_pct.index >= START_DATE]

    # 加载ETF月度收益
    print("加载ETF收益数据...")
    close = load_close_matrix()
    monthly_close = close.resample("ME").last()
    etf_ret = monthly_close.pct_change()

    # 计算 IC（行业ETF）
    print("\n计算截面 Rank IC（行业ETF）...")
    ic_1m_sector  = calc_cross_section_ic(flow_1m,     etf_ret, sector_valid)
    ic_3m_sector  = calc_cross_section_ic(flow_3m_pct, etf_ret, sector_valid)

    print("\n--- 行业ETF ---")
    print_ic_stats(ic_1m_sector,  "单月资金流变化率")
    print_ic_stats(ic_3m_sector,  "3月资金流历史分位数")

    # 全ETF（含宽基）
    print("\n计算截面 Rank IC（全部ETF含宽基）...")
    ic_1m_all = calc_cross_section_ic(flow_1m,     etf_ret, all_valid)
    ic_3m_all = calc_cross_section_ic(flow_3m_pct, etf_ret, all_valid)

    print("\n--- 全部ETF（含宽基）---")
    print_ic_stats(ic_1m_all, "单月资金流变化率")
    print_ic_stats(ic_3m_all, "3月资金流历史分位数")

    # 分组收益（行业ETF，Q1=净流出最多）
    print("\n分组收益分析（行业ETF，Q1=净流出最多）：")
    print("\n单月资金流变化率：")
    qret_1m = calc_quintile_returns(flow_1m, etf_ret, sector_valid)
    if not qret_1m.empty:
        print(qret_1m.to_string())
    print("\n3月资金流历史分位数：")
    qret_3m = calc_quintile_returns(flow_3m_pct, etf_ret, sector_valid)
    if not qret_3m.empty:
        print(qret_3m.to_string())

    # 绘图
    print("\n生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ETF 资金净流量 IC 验证（反向信号假设）", fontsize=13, fontweight="bold")

    for ax, ic, title in [
        (axes[0, 0], ic_1m_sector, "单月资金流变化率 IC（行业ETF）"),
        (axes[0, 1], ic_3m_sector, "3月资金流分位数 IC（行业ETF）"),
        (axes[1, 0], ic_1m_all,    "单月资金流变化率 IC（全部ETF）"),
        (axes[1, 1], ic_3m_all,    "3月资金流分位数 IC（全部ETF）"),
    ]:
        if not ic.empty:
            ic.plot(ax=ax, color="steelblue", alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.8)
            mean_val = ic.mean()
            ax.axhline(mean_val, color="red", linestyle="--",
                       label=f"均值={mean_val:.3f}")
            ax.fill_between(ic.index, ic, 0,
                            where=ic > 0, alpha=0.3, color="green")
            ax.fill_between(ic.index, ic, 0,
                            where=ic <= 0, alpha=0.3, color="red")
            ax.set_title(title)
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"{title}（无数据）")

    plt.tight_layout()
    out_path = RESULTS_DIR / "fund_flow_ic_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{out_path}")

    # 结论
    print("\n" + "=" * 60)
    print("结论判断标准：")
    print("  IC均值 < -0.05 且 IC<0占比 > 55%  → 反向信号有效（净流出→次月涨）")
    print("  IC均值 > +0.05 且 IC>0占比 > 55%  → 同向信号有效（净流入→次月涨）")
    print("  |IC均值| < 0.03                    → 无效")
    print("=" * 60)

    for ic, label in [
        (ic_1m_sector, "单月流量（行业ETF）"),
        (ic_3m_sector, "3月分位数（行业ETF）"),
    ]:
        if ic.empty:
            continue
        ic_mean = ic.mean()
        pos = (ic > 0).mean()
        if ic_mean < -0.05 and pos < 0.45:
            verdict = "反向信号有效 — 净流出为买入信号"
        elif ic_mean > 0.05 and pos > 0.55:
            verdict = "同向信号有效 — 净流入为买入信号"
        elif abs(ic_mean) > 0.03:
            verdict = "弱有效，方向需确认"
        else:
            verdict = "无效"
        print(f"\n{label}: IC均值={ic_mean:.4f}，IC>0占比={pos:.1%} → {verdict}")


if __name__ == "__main__":
    main()
