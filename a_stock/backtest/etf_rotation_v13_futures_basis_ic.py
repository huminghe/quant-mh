"""
第十一轮方向2：股指期货基差（贴水/升水）作为市场状态信号 IC 检验

背景：已测的市场状态过滤全部基于现货价格/估值（大盘MA200/PE分位区制切换），
均因"高波动=A股趋势最强期"这一机制系统性失效（见 lessons.md）。股指期货
基差反映的是衍生品市场对未来的隐含预期（深度贴水通常对应恐慌/去杠杆），
信号来源与现货价格趋势不同，值得单独检验是否踩中同一个坑。

方法（先IC检验排除法）：
1. 用 IF/IH/IC 主力连续合约（沪深300/上证50/中证500对应）与现货指数计算基差
2. 基差水平、基差变化 两个版本，分别与未来1月沪深300收益做IC检验
3. 检验基差信号是否与"大盘MA200过滤"同源（本质都是判断市场状态），
   若信号在样本期内的失效模式相同（高波动期=趋势最强期时基差也深度贴水导致踩空），
   直接判定与已测方向同类失败，不需要进组合回测
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro

START_DATE = "2016-01-01"
END_DATE = "2026-07-10"

FUTURES = {"IF.CFX": "000300.SH", "IH.CFX": "000016.SH", "IC.CFX": "000905.SH"}
NAMES = {"IF.CFX": "沪深300股指期货", "IH.CFX": "上证50股指期货", "IC.CFX": "中证500股指期货"}


def fetch_fut_daily_full(pro, ts_code: str) -> pd.DataFrame:
    """fut_daily 单次调用约2000条截断，按年分段拉取绕过。"""
    start_year = int(START_DATE[:4])
    end_year = int(END_DATE[:4])
    chunks = []
    for y in range(start_year, end_year + 1):
        df = pro.fut_daily(ts_code=ts_code, start_date=f"{y}0101", end_date=f"{y}1231")
        if df is not None and not df.empty:
            chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    full = pd.concat(chunks, ignore_index=True).drop_duplicates("trade_date").sort_values("trade_date")
    full["trade_date"] = pd.to_datetime(full["trade_date"])
    return full.set_index("trade_date")[["close"]].rename(columns={"close": "fut_close"})


def fetch_index_daily(pro, ts_code: str) -> pd.DataFrame:
    df = pro.index_daily(ts_code=ts_code, start_date=START_DATE.replace("-", ""), end_date=END_DATE.replace("-", ""))
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").set_index("trade_date")[["close"]].rename(columns={"close": "idx_close"})


print("从 tushare 拉取股指期货主力连续合约 + 现货指数数据...")
pro = init_pro()

basis_data = {}
for fut_code, idx_code in FUTURES.items():
    fut = fetch_fut_daily_full(pro, fut_code)
    idx = fetch_index_daily(pro, idx_code)
    merged = fut.join(idx, how="inner").dropna()
    merged["basis"] = merged["fut_close"] / merged["idx_close"] - 1
    basis_data[fut_code] = merged
    print(f"  {fut_code}（{NAMES[fut_code]}）：{len(merged)} 个交易日，"
          f"{merged.index[0].date()} ~ {merged.index[-1].date()}，"
          f"基差均值={merged['basis'].mean():+.4f}，std={merged['basis'].std():.4f}")

# ── IC检验：用IF基差（沪深300对应，覆盖最广）预测沪深300未来1月收益 ─────

print("\n" + "=" * 80)
print("诊断1：IF.CFX 基差（贴水/升水）vs 沪深300未来1月收益 IC 检验")
print("=" * 80)

hs300 = basis_data["IF.CFX"][["idx_close"]].copy()
hs300["fwd_1m"] = hs300["idx_close"].pct_change(21).shift(-21)
hs300["basis"] = basis_data["IF.CFX"]["basis"]
hs300["basis_chg_5d"] = hs300["basis"].diff(5)  # 基差5日变化，捕捉恐慌加速

for sig_col, sig_name in [("basis", "基差水平"), ("basis_chg_5d", "基差5日变化")]:
    df = hs300[[sig_col, "fwd_1m"]].dropna()
    ic = df[sig_col].corr(df["fwd_1m"])
    hit = (np.sign(df[sig_col]) == np.sign(df["fwd_1m"])).mean()

    yearly_ics = []
    for yr in sorted(set(df.index.year)):
        seg = df[df.index.year == yr]
        if len(seg) < 20:
            continue
        yearly_ics.append((yr, seg[sig_col].corr(seg["fwd_1m"])))
    pos_ratio = np.mean([1 if v > 0 else 0 for _, v in yearly_ics]) if yearly_ics else np.nan

    print(f"\n{sig_name}：全样本IC={ic:+.4f}，方向命中率={hit:.1%}，样本数={len(df)}")
    print(f"  年度IC：{[(y, round(v, 3)) for y, v in yearly_ics]}")
    print(f"  年度同向占比：{pos_ratio:.1%}")

# ── 诊断2：深度贴水年份 vs 大盘MA200过滤失效年份是否重叠（同源检验）───

print("\n" + "=" * 80)
print("诊断2：深度贴水（basis < 25分位）年份分布，对比是否与'高波动=趋势最强期'冲突")
print("=" * 80)

basis_25pct = hs300["basis"].quantile(0.25)
deep_backwardation = hs300[hs300["basis"] < basis_25pct]
year_counts = deep_backwardation.groupby(deep_backwardation.index.year).size()
print(f"基差25分位阈值：{basis_25pct:.4f}")
print(f"深度贴水交易日的年度分布：\n{year_counts.to_string()}")

# 深度贴水期间的沪深300未来1月收益（若为负，说明贴水确实领先下跌；若为正/混合，说明贴水期恰好是反弹期）
dp_fwd = deep_backwardation["fwd_1m"].dropna()
print(f"\n深度贴水期间沪深300未来1月收益：均值={dp_fwd.mean():+.2%}，"
      f"正收益占比={((dp_fwd > 0).mean()):.1%}，样本数={len(dp_fwd)}")

# ── 结论 ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
level_ic = hs300[["basis", "fwd_1m"]].dropna()["basis"].corr(hs300[["basis", "fwd_1m"]].dropna()["fwd_1m"])
chg_ic = hs300[["basis_chg_5d", "fwd_1m"]].dropna()["basis_chg_5d"].corr(
    hs300[["basis_chg_5d", "fwd_1m"]].dropna()["fwd_1m"])
print(f"基差水平IC={level_ic:+.4f}，基差变化IC={chg_ic:+.4f}")
print(f"深度贴水期间未来1月收益均值={dp_fwd.mean():+.2%}（若为正，说明贴水信号在A股会导致"
      f"'恐慌买入点误判为风险规避点'，与MA200/PE区制切换同源失效）")

if max(abs(level_ic), abs(chg_ic)) < 0.03:
    print("判定：IC绝对值 < 0.03，信号强度不足，不建议进入组合层面回测。")
elif dp_fwd.mean() > 0:
    print("判定：深度贴水期间未来收益反而为正，信号会在'恐慌买点'误判为风险规避，"
          "与已测的PE区制切换/MA200过滤同源失效模式一致，不建议进入组合层面回测。")
else:
    print("判定：IC强度、方向一致性通过初筛，且未发现与已测方向同源失效的证据，"
          "值得进入组合层面回测验证。")

print("\n完成。")
