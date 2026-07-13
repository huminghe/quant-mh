"""
第十一轮方向3：跨市场溢出（隔夜美股/港股收益）作为板块轮动信号 IC 检验

背景：已测的信号全部基于A股自身价格/估值/衍生品，从未引入跨市场信息。
A股收盘（15:00）早于美股开盘（21:30北京时间），隔夜美股/港股涨跌是A股次日
开盘前唯一可用的"新信息"，海外文献对亚洲市场的隔夜溢出效应有稳定记录。
标的池里QDII（纳指/标普500/恒生/恒生科技/港股互联网/中概互联）本身就与海外
市场高度相关，这里检验的是"隔夜海外收益能否预测A股内需/成长板块（计算机/
半导体/新能源等）次日表现"，与QDII标的自身动量是不同的信息（QDII已在轮动
池里吃到了海外趋势，这里测的是海外→内需板块的溢出，不同传导路径）。

方法（先IC检验排除法）：
1. 计算隔夜标普500/纳指/恒生指数收益（T-1收盘到T-1收盘，即A股T日开盘前
   最新可见的海外收益）
2. 与A股次日（T日）沪深300、成长风格板块（用半导体/新能源/计算机行业ETF
   等风险资产代表）收益做IC检验
3. 检验该信号是否与QDII标的自身动量高度重叠（若QDII已经吃到了这个信息，
   单独引入意义不大）
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro, load_close_matrix

START_DATE = "2016-01-01"
END_DATE = "2026-07-10"

GLOBAL_INDICES = {"SPX": "标普500", "IXIC": "纳斯达克综指", "HSI": "恒生指数"}

# A股风险资产代表（成长/内需板块，非QDII本身）
RISK_SECTOR_ETFS = {
    "515000.SH": "计算机ETF",
    "512760.SH": "半导体ETF",
    "515330.SH": "新能源车ETF",
    "159928.SZ": "消费ETF",
}


def fetch_global_daily(pro, code: str) -> pd.Series:
    df = pro.index_global(ts_code=code, start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""))
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df["close"].pct_change().rename(code)


print("从 tushare 拉取海外指数 + A股行业ETF数据...")
pro = init_pro()

overnight_rets = {}
for code, name in GLOBAL_INDICES.items():
    ret = fetch_global_daily(pro, code)
    overnight_rets[code] = ret
    print(f"  {code}（{name}）：{len(ret.dropna())} 个交易日收益，"
          f"{ret.index[0].date()} ~ {ret.index[-1].date()}")

overnight_df = pd.DataFrame(overnight_rets)

close_matrix = load_close_matrix(list(RISK_SECTOR_ETFS.keys()))
astock_rets = close_matrix.pct_change()

# ── 对齐时区：海外指数 trade_date=T-1（美股/港股收盘日），A股 T-1 收盘到 T 收盘
# 即用"海外T-1收益"预测"A股T日收益"，因为海外收盘略晚于A股，T-1的海外收益是
# A股T日开盘前最新可见信息 ───────────────────────────────────────────────

print("\n" + "=" * 80)
print("诊断1：隔夜海外收益 vs A股次日行业ETF收益 IC 检验")
print("=" * 80)

astock_dates = astock_rets.index

results = []
for global_code in GLOBAL_INDICES:
    overnight = overnight_df[global_code].dropna()
    # 把海外收益的日期索引对齐到"下一个A股交易日"：海外T-1的收益，映射到A股中
    # 第一个 > T-1 的交易日
    for etf_code, etf_name in RISK_SECTOR_ETFS.items():
        etf_ret = astock_rets[etf_code].dropna()
        # 构造对齐序列：对每个A股交易日T，找最近的一个海外交易日（<T）的隔夜收益
        aligned = pd.Series(index=etf_ret.index, dtype=float)
        overnight_dates = overnight.index
        for i, d in enumerate(etf_ret.index):
            prior = overnight_dates[overnight_dates < d]
            if len(prior) == 0:
                continue
            aligned.iloc[i] = overnight.loc[prior[-1]]
        merged = pd.DataFrame({"overnight": aligned, "fwd_ret": etf_ret}).dropna()
        if len(merged) < 100:
            continue
        ic = merged["overnight"].corr(merged["fwd_ret"])
        hit = (np.sign(merged["overnight"]) == np.sign(merged["fwd_ret"])).mean()

        yearly_ics = []
        for yr in sorted(set(merged.index.year)):
            seg = merged[merged.index.year == yr]
            if len(seg) < 20:
                continue
            yearly_ics.append((yr, seg["overnight"].corr(seg["fwd_ret"])))
        pos_ratio = np.mean([1 if v > 0 else 0 for _, v in yearly_ics]) if yearly_ics else np.nan

        results.append({
            "海外指数": global_code, "A股ETF": etf_name, "IC": ic, "方向命中率": hit,
            "年度同向占比": pos_ratio, "样本数": len(merged),
        })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

best = res_df.loc[res_df["IC"].abs().idxmax()]
print(f"\n最强信号：{best['海外指数']} → {best['A股ETF']}，IC={best['IC']:+.4f}，"
      f"年度同向占比={best['年度同向占比']:.1%}")

# ── 诊断2：与QDII标的自身动量的重叠度检验（用恒生ETF 159920 次日收益近似"QDII已吃到的信息"）
print("\n" + "=" * 80)
print("诊断2：隔夜恒生指数收益 vs 恒生ETF(159920)自身次日收益 —— 检验QDII是否已吃到该信息")
print("=" * 80)

hsi_overnight = overnight_df["HSI"].dropna()
hs_etf = load_close_matrix(["159920.SZ"])["159920.SZ"].pct_change().dropna()
aligned_hsi = pd.Series(index=hs_etf.index, dtype=float)
for i, d in enumerate(hs_etf.index):
    prior = hsi_overnight.index[hsi_overnight.index < d]
    if len(prior) == 0:
        continue
    aligned_hsi.iloc[i] = hsi_overnight.loc[prior[-1]]
merged_hsi = pd.DataFrame({"overnight": aligned_hsi, "own_ret": hs_etf}).dropna()
ic_qdii = merged_hsi["overnight"].corr(merged_hsi["own_ret"])
print(f"隔夜恒生收益 vs 恒生ETF自身次日收益 IC={ic_qdii:+.4f}（样本数={len(merged_hsi)}）")
print("（若该IC远高于诊断1中'海外→内需板块'的溢出IC，说明QDII标的本身已充分吃到该信息，"
      "单独引入内需板块溢出信号的增量价值需要看两者的差距是否足够大）")

# ── 结论 ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
max_ic = res_df["IC"].abs().max()
print(f"跨市场溢出信号最强IC={max_ic:+.4f}（阈值0.03）")
if max_ic < 0.03:
    print("判定：IC绝对值 < 0.03，信号强度不足，不建议进入组合层面回测。")
elif best["年度同向占比"] < 0.6:
    print(f"判定：年度同向占比{best['年度同向占比']:.1%} < 60%，方向不稳定，不建议进入组合层面回测。")
else:
    print("判定：IC强度、方向一致性通过初筛，值得进入组合层面回测验证。")

print("\n完成。")
