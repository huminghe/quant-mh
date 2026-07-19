"""
申万一级行业分层抽样候选池——用系统化规则替代研究员手工挑选行业。

背景：2026-07-15/17已确认45只手工圈定标的池夏普1.053，换成纯流动性排名的
机械化候选池（431只）后降到0.59，且这个差距不能靠叠加ML信号集成弥补。
诊断结论是1.053里含有"研究员事后知道哪些行业涨得好"这个人工判断（前视偏差）。

本脚本用 build_etf_sw_exposure.py 算出的ETF→申万一级行业暴露分类结果，
按行业分层挑选代表ETF（每行业1-2只，按流动性），叠加客观识别的宽基篮子（Top5），
构建一个不依赖人工主题判断的候选池，测试能否逼近1.053夏普。

第一版简化：行业分类用ETF最新一期持仓做一次性分类（非全PIT滚动），
候选池成员在整个回测区间固定不变。这个简化本身的前视偏差（用当前持仓结构
判断历史时点的行业归属）比45只手工池的前视偏差（用最终收益表现挑选品种）更轻，
但在结论中需要明确标注这个局限。
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import (  # noqa: E402
    calc_all_scores, get_rebalance_dates, run_backtest, calc_stats,
    MOMENTUM_WINDOW, START_DATE, CASH_ETF,
)
from etf_rotation_v23_universe_bias_test import (  # noqa: E402
    CACHE_DIR, load_close_matrix_from_cache,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
EXPOSURE_PATH = DATA_DIR / "etf_sw_exposure.parquet"
TURNOVER_PATH = DATA_DIR / "market_turnover.parquet"
META_PATH = DATA_DIR / "market_etf_meta.parquet"

TOP_K_BROAD = 5           # 宽基篮子保留只数
SECOND_PLACE_RATIO = 0.5  # 行业内第2名成交额>=第1名此比例则两只都保留


def latest_liquidity(codes: list) -> pd.Series:
    """按候选代码计算最近126个交易日日均成交额（千元），用于流动性排序。"""
    turnover = pd.read_parquet(TURNOVER_PATH)
    turnover = turnover[turnover["ts_code"].isin(codes)]
    wide = turnover.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    recent_avg = wide.tail(126).mean()
    return recent_avg.reindex(codes)


def build_stratified_pool(summary: pd.DataFrame, liquidity: pd.Series) -> tuple:
    """按申万一级行业分层选代表ETF + 客观识别的宽基篮子，返回(最终候选池, 诊断信息dict)。"""
    summary = summary.copy()
    summary["liquidity"] = summary["ts_code"].map(liquidity)
    summary = summary[summary["liquidity"].notna()]  # 无成交额数据的候选（如新上市）排除

    industry_reps = {}
    for industry, group in summary[summary["classification"] == "industry"].groupby("dominant_industry"):
        ranked = group.sort_values("liquidity", ascending=False)
        picks = [ranked.iloc[0]["ts_code"]]
        if len(ranked) > 1 and ranked.iloc[1]["liquidity"] >= ranked.iloc[0]["liquidity"] * SECOND_PLACE_RATIO:
            picks.append(ranked.iloc[1]["ts_code"])
        industry_reps[industry] = picks

    broad_pool = summary[summary["classification"].isin(["broad", "broad_qdii"])]
    broad_top = broad_pool.sort_values("liquidity", ascending=False).head(TOP_K_BROAD)["ts_code"].tolist()

    industry_codes = sorted({c for picks in industry_reps.values() for c in picks})
    final_pool = sorted(set(industry_codes) | set(broad_top))

    diagnostics = {
        "industry_reps": industry_reps,
        "broad_top": broad_top,
        "n_industry_codes": len(industry_codes),
        "n_broad_codes": len(broad_top),
        "n_final": len(final_pool),
    }
    return final_pool, diagnostics


def print_diagnostics(diagnostics: dict, summary: pd.DataFrame, meta: pd.DataFrame):
    name_map = meta.set_index("ts_code")["name"]

    print("\n" + "=" * 60)
    print(f"分层候选池构建结果：{diagnostics['n_final']} 只"
          f"（行业代表 {diagnostics['n_industry_codes']} + 宽基 {diagnostics['n_broad_codes']}）")
    print("=" * 60)

    print(f"\n覆盖申万一级行业数：{len(diagnostics['industry_reps'])} / 31")
    print("\n各行业代表ETF清单：")
    for industry, codes in sorted(diagnostics["industry_reps"].items()):
        for code in codes:
            row = summary[summary["ts_code"] == code].iloc[0]
            name = name_map.get(code, "?")
            print(f"  {industry:6s} | {code} {name} 集中度={row['concentration']:.3f}")

    print("\n宽基篮子清单：")
    for code in diagnostics["broad_top"]:
        name = name_map.get(code, "?")
        row = summary[summary["ts_code"] == code].iloc[0]
        print(f"  {code} {name} classification={row['classification']}")

    all_sw = set(pd.read_parquet(DATA_DIR / "stock_sw_industry.parquet")["sw_industry"].unique())
    empty_industries = all_sw - set(diagnostics["industry_reps"].keys())
    print(f"\n空行业（无合格代表ETF，共{len(empty_industries)}个）：{sorted(empty_industries)}")


def main():
    print("加载ETF行业暴露分类结果...")
    summary = pd.read_parquet(EXPOSURE_PATH)
    meta = pd.read_parquet(META_PATH)

    candidates = summary["ts_code"].tolist()
    print(f"候选池共 {len(candidates)} 只，计算最近126日流动性...")
    liquidity = latest_liquidity(candidates)

    final_pool, diagnostics = build_stratified_pool(summary, liquidity)
    print_diagnostics(diagnostics, summary, meta)

    print("\n加载价格矩阵（复用v23价格缓存）...")
    close_full = load_close_matrix_from_cache(final_pool)
    close = close_full[close_full.index >= START_DATE]
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]
    dropped = set(final_pool) - set(valid_codes)
    if dropped:
        print(f"警告：{len(dropped)}只候选无足够价格历史被剔除：{sorted(dropped)}")
    print(f"最终参与回测标的数：{len(valid_codes)}")

    print(f"\n计算动量得分（窗口={MOMENTUM_WINDOW}日，风险调整）...")
    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)
    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]

    print("运行回测...")
    nav = run_backtest(close, scores, rebal_dates, cash_etf=CASH_ETF)

    print("\n" + "=" * 60)
    print("申万行业分层候选池回测结果")
    print("=" * 60)
    print(f"回测区间：{nav.index[0].date()} → {nav.index[-1].date()}")
    stats = calc_stats(nav, f"申万分层候选池({len(valid_codes)}只)")
    print(pd.DataFrame([stats]).set_index("标的").to_string())

    print("\n对照表：")
    print("  45只手工标的池·纯动量：夏普 1.053")
    print("  机械化候选池(431只)·纯动量：夏普 0.59")
    print("  机械化候选池·Top100上限·纯动量：夏普 0.53")
    print(f"  申万行业分层候选池({len(valid_codes)}只)·纯动量：夏普 {stats['年化夏普']}（本次）")


if __name__ == "__main__":
    main()
