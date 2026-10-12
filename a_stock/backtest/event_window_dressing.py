"""
公募基金季末window dressing价格异动效应诊断（事件研究，前提检验）

背景：第46轮候选盘点后子agent建议的"确定性事件+被动资金流"子方向里，
唯一尚未被测试过的候选（ETF份额申赎冲击已证伪见第十二轮②，持仓增减仓强度
因子已证伪见第十六轮①）。详见 docs/research_index_enhancement.md 第四十六轮
背景段落。

信号逻辑：基金经理在季度末（3/6/9/12月最后交易日）有做高净值报告期表现的
动机，倾向于集中买入前期强势股拉升收盘价（"做仓"），季末后该拉升效应回吐，
表现为季末前涨幅与季末后收益负相关（反转）。这不依赖fund_portfolio持仓
数据（Q1/Q3只披露前十大重仓，覆盖口径不一致，见第十六轮①的数据可行性
结论），只用stock_daily价格数据，可覆盖全部4个季末，不止Q2/Q4两个。

方法：沿用 event_index_rebalance.py 的"先测前提，不显著直接判定证伪"
方法论——每个季末窗口，测"窗口前N日涨跌幅"（因子）与"窗口后N日收益"
（前瞻收益）的横截面Spearman Rank IC。若window dressing存在，预期IC为
负（赛季末前涨幅大的股票，季末后出现回吐式下跌）。

用法：
  cd a_stock/backtest
  python event_window_dressing.py               # 默认中证500
  python event_window_dressing.py --index hs300
"""

import sys
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import load_close_panel, load_members_pit, DATA_DIR

START_DATE = "2016-01-01"
END_DATE = "2026-06-30"
MIN_STOCKS_PER_CROSS = 50

PRE_WINDOW_DAYS = 5   # 季末前N个交易日涨跌幅（因子）
POST_WINDOW_DAYS = 5  # 季末后N个交易日收益（前瞻收益）

INDEX_CONFIG = {
    "hs300": {"name": "沪深300", "members_file": DATA_DIR / "hs300_members.parquet"},
    "hs500": {"name": "中证500", "members_file": DATA_DIR / "hs500_members.parquet"},
}


def winsorize(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def cross_section_rank_ic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret"]
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
    return ic


def get_season_ends(trade_days: pd.DatetimeIndex) -> list:
    """每个自然季度(3/6/9/12月)的最后一个真实交易日"""
    df = pd.Series(trade_days, index=trade_days)
    quarter_key = df.index.to_series().dt.to_period("Q")
    return sorted(df.groupby(quarter_key).apply(lambda x: x.index.max()).tolist())


def compute_window_dressing_ic(close_panel: pd.DataFrame, members_file: pathlib.Path) -> pd.DataFrame:
    close_panel = close_panel.loc[START_DATE:END_DATE]
    trade_days = close_panel.index
    season_ends = get_season_ends(trade_days)

    records = []
    for q_end in season_ends:
        pos = trade_days.get_indexer([q_end])[0]
        pre_pos = pos - PRE_WINDOW_DAYS
        post_pos = pos + POST_WINDOW_DAYS
        if pre_pos < 0 or post_pos >= len(trade_days):
            continue
        pre_date, post_date = trade_days[pre_pos], trade_days[post_pos]

        pit_members = load_members_pit(q_end, members_file=members_file)
        if not pit_members:
            continue
        available = [c for c in pit_members if c in close_panel.columns]
        if len(available) < MIN_STOCKS_PER_CROSS:
            continue

        pre_price = close_panel[available].loc[pre_date].dropna()
        end_price = close_panel[available].loc[q_end].dropna()
        post_price = close_panel[available].loc[post_date].dropna()

        common = pre_price.index.intersection(end_price.index).intersection(post_price.index)
        if len(common) < MIN_STOCKS_PER_CROSS:
            continue

        factor = end_price[common] / pre_price[common] - 1  # 季末前涨跌幅
        fwd_ret = post_price[common] / end_price[common] - 1  # 季末后收益
        factor = winsorize(factor)

        ic = cross_section_rank_ic(factor, fwd_ret)
        records.append({"date": q_end, "ic": ic, "n_stocks": len(common)})

    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()


def summarize_ic(ic_series: pd.Series) -> dict:
    clean = ic_series.dropna()
    overall_mean = clean.mean()
    same_sign = (np.sign(clean) == np.sign(overall_mean)).mean() if overall_mean != 0 and len(clean) else 0.0
    passed = abs(overall_mean) >= 0.03 and same_sign >= 0.6
    return {
        "样本季度数": len(clean),
        "IC均值": round(overall_mean, 4),
        "IC标准差": round(clean.std(), 4),
        "ICIR": round(overall_mean / clean.std(), 3) if clean.std() > 0 else np.nan,
        "同向占比": f"{same_sign * 100:.1f}%",
        "通过前提检验": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="季末window dressing价格异动效应诊断")
    parser.add_argument("--index", choices=["hs300", "hs500"], default="hs500")
    args = parser.parse_args()

    cfg = INDEX_CONFIG[args.index]
    print(f"=== 季末window dressing效应诊断：{cfg['name']} ===")
    print(f"窗口：季末前{PRE_WINDOW_DAYS}日涨跌幅 → 季末后{POST_WINDOW_DAYS}日收益\n")

    close_panel = load_close_panel()
    ic_df = compute_window_dressing_ic(close_panel, cfg["members_file"])

    if ic_df.empty:
        print("无有效样本季度，诊断终止。")
        return

    print(ic_df.to_string())
    stats = summarize_ic(ic_df["ic"])
    print("\n汇总统计：")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
