"""
新股/次新股上市初期资金流效应诊断（事件研究，仅测核心前提）

背景：指数增强第十一轮候选①（a_stock/docs/research.md「指数增强策略」
第十一轮小节）。逻辑：新股上市首周的资金承接强度（首日涨幅、首周累计涨停
天数、首周平均成交额）代表市场对该股的追捧程度，这部分资金是被动指数基金
无法参与的增量资金（新股纳入指数需等待一定观察期）。假设：资金承接越强，
后续（脱离炒新泡沫后）跌得越多——这是"炒新反转"假设，与"资金承接强预测
后续继续走强"的"新股强者恒强"假设方向相反，本脚本先不预设方向，只测
分组的截面IC和高低分组的超额收益差，看数据本身指向哪个方向。

数据与方法：
- 事件池：new_share.parquet（fetch_new_share.py），限定issue_date非空
  （已上市），2016-2026年全部新股（不限沪深300/中证500股票池——新股
  上市初期还没被纳入任何宽基指数成分股，用现有hs300/hs500池子会把
  样本过滤到几乎为空，这里的基准也不用个股所属指数，统一用中证800）。
- 上市首周资金流代理指标（用new_share_daily_window.parquet，
  fetch_new_share_window.py 拉取的上市后120自然日窗口日线）：
  - day1_pct_chg：上市首日涨幅
  - week1_limit_up_days：上市后前5个交易日涨停天数（pct_chg>=9.5%近似，
    新股上市首日不受涨跌停限制但普遍首日大涨，第2天起才受限，这里统计
    第2-5个交易日）
  - week1_avg_amount：上市后前5个交易日平均成交额（元）
- 建仓时点：**不能固定用"第6个交易日"**（已实测踩坑：33%的新股在第6个
  交易日仍处于一字涨停锁死状态，open==close且涨幅>=9.5%，根本无法按
  开盘价买入；连续锁死天数中位数为0但75分位数达7天、最大28天，波动
  极大，固定偏移量必然把大量事件的"建仓价"设在不可执行的涨停价上，
  产生虚假暴利——这是本轮第二次踩到"建仓机制忽略涨跌停可执行性导致
  假显著"的坑，第一次是event_index_rebalance.py的T+1建仓，这里的
  新增教训是"事件驱动信号如果预期首周有持续涨停，必须先扫描锁死区间
  再确定建仓日，不能直接套用固定offset模板"）。改为：从第2个交易日
  起扫描，找到第一个未锁死（open!=close或涨幅<9.5%）的交易日作为
  建仓日；若120自然日窗口内始终锁死，该事件剔除（真实约束，不是
  数据缺陷，代表该新股散户全程无法参与炒作只能在打新阶段获利）。
- 测试窗口：20/60个交易日累计超额收益（先测短窗口看衰减速度，符合
  .claude/lessons.md第99条方法论）。
- 基准：中证800（000906.SH）。
- 分组方法：按week1_avg_amount（首周资金承接强度代理）分5组，看高低组
  超额收益差是否单调、是否显著（截面分组测试，比单一相关系数更直观地
  看信号方向和非线性）。

复用 event_index_rebalance.py 的通用工具函数（T+1建仓价/退出价、涨跌停
可执行性检查、成本口径），避免重复实现同一套方法论（DRY）。

用法：
  cd a_stock/backtest
  python event_new_share_flow.py
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import init_pro, DATA_DIR
from event_index_rebalance import (
    shift_trading_day, exit_price, load_index_daily, index_window_return,
    ROUND_TRIP_COST,
)

NEW_SHARE_FILE = DATA_DIR / "new_share.parquet"
WINDOW_FILE = DATA_DIR / "new_share_daily_window.parquet"
BENCHMARK_CODE = "000906.SH"  # 中证800

HOLD_WINDOWS = [20, 60]
N_GROUPS = 5
LIMIT_UP_THRESHOLD = 0.095  # 涨停判断阈值（近似，含缓冲）


def load_new_share() -> pd.DataFrame:
    if not NEW_SHARE_FILE.exists():
        raise FileNotFoundError(f"缺少 {NEW_SHARE_FILE}，请先运行 fetch_new_share.py")
    df = pd.read_parquet(NEW_SHARE_FILE)
    return df.dropna(subset=["issue_date"]).copy()


def load_window() -> pd.DataFrame:
    if not WINDOW_FILE.exists():
        raise FileNotFoundError(f"缺少 {WINDOW_FILE}，请先运行 fetch_new_share_window.py")
    df = pd.read_parquet(WINDOW_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def find_first_unlocked_entry(g: pd.DataFrame) -> tuple:
    """从第2个交易日（0-indexed第1行）起扫描，返回第一个未一字涨停锁死交易日的
    (trade_date, open价)；若窗口内始终锁死则返回(None, None)"""
    for i in range(1, len(g)):
        r = g.iloc[i]
        locked = (abs(r["open"] - r["close"]) < 1e-6) and (r["pct_chg"] / 100.0 >= LIMIT_UP_THRESHOLD)
        if not locked:
            return r["trade_date"], r["open"]
    return None, None


def build_flow_features(new_share: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """对每只新股计算首周资金流代理指标 + 首个可执行建仓日"""
    rows = []
    n_all_locked = 0
    for _, row in new_share.iterrows():
        ts_code = row["ts_code"]
        g = window[window["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        if len(g) < 6:
            continue
        day1_pct_chg = g.iloc[0]["pct_chg"] / 100.0
        week1 = g.iloc[1:5]  # 第2-5个交易日
        if week1.empty:
            continue
        week1_limit_up_days = int((week1["pct_chg"] / 100.0 >= LIMIT_UP_THRESHOLD).sum())
        week1_avg_amount = week1["amount"].mean() * 1000  # amount单位千元，转元
        entry_date, entry_price_val = find_first_unlocked_entry(g)
        if entry_date is None:
            n_all_locked += 1
            continue
        rows.append({
            "ts_code": ts_code, "issue_date": row["issue_date"],
            "day1_pct_chg": day1_pct_chg,
            "week1_limit_up_days": week1_limit_up_days,
            "week1_avg_amount": week1_avg_amount,
            "entry_date": entry_date, "entry_price": entry_price_val,
        })
    print(f"窗口内始终一字涨停锁死（无法参与，已剔除）：{n_all_locked} 只")
    return pd.DataFrame(rows)


def net_return_analysis(features: pd.DataFrame, trade_days: pd.DatetimeIndex,
                         index_close: pd.Series) -> pd.DataFrame:
    rows = []
    for _, ev in features.iterrows():
        entry_date = ev["entry_date"]
        p_in = ev["entry_price"]
        if pd.isna(p_in) or p_in <= 0:
            continue
        for window_days in HOLD_WINDOWS:
            exit_date = shift_trading_day(trade_days, entry_date, window_days)
            if exit_date is None:
                continue
            p_out = exit_price(ev["ts_code"], exit_date)
            if pd.isna(p_out):
                continue
            stock_ret = p_out / p_in - 1
            idx_ret = index_window_return(index_close, entry_date, exit_date)
            if pd.isna(idx_ret):
                continue
            gross_excess = stock_ret - idx_ret
            net_excess = gross_excess - ROUND_TRIP_COST
            rows.append({
                "ts_code": ev["ts_code"], "entry_date": entry_date.date(),
                "window": window_days,
                "week1_avg_amount": ev["week1_avg_amount"],
                "day1_pct_chg": ev["day1_pct_chg"],
                "week1_limit_up_days": ev["week1_limit_up_days"],
                "gross_excess": gross_excess, "net_excess": net_excess,
            })
    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> None:
    print("\n=== 全样本净收益核算（扣完整回合成本%.3f%%）===" % (ROUND_TRIP_COST * 100))
    for window_days in HOLD_WINDOWS:
        sub = df[df["window"] == window_days]["net_excess"].dropna()
        if sub.empty:
            print(f"\n持有{window_days}个交易日：无有效数据")
            continue
        mean = sub.mean()
        same_sign = (np.sign(sub) == np.sign(mean)).mean() if mean != 0 else 0.0
        t_stat, p_val = stats.ttest_1samp(sub, 0)
        print(f"持有{window_days}个交易日：净超额收益均值={mean:+.4%}  "
              f"同向占比={same_sign:.1%}  n={len(sub)}事件  t={t_stat:.2f}  p={p_val:.3f}")


def summarize_groups(df: pd.DataFrame, factor_col: str) -> None:
    print(f"\n=== 按{factor_col}分{N_GROUPS}组（组1=最低，组{N_GROUPS}=最高）===")
    for window_days in HOLD_WINDOWS:
        sub = df[df["window"] == window_days].dropna(subset=[factor_col, "net_excess"]).copy()
        if len(sub) < N_GROUPS * 10:
            print(f"\n持有{window_days}个交易日：样本不足，跳过分组")
            continue
        sub["group"] = pd.qcut(sub[factor_col], N_GROUPS, labels=False, duplicates="drop") + 1
        summary = sub.groupby("group")["net_excess"].agg(["mean", "count"])
        print(f"\n持有{window_days}个交易日：")
        print(summary.to_string())
        if summary.index.max() >= 2:
            spread = summary.loc[summary.index.max(), "mean"] - summary.loc[summary.index.min(), "mean"]
            print(f"  高低组差（组{summary.index.max()}-组{summary.index.min()}）= {spread:+.4%}")
        ic, ic_p = stats.spearmanr(sub[factor_col], sub["net_excess"])
        print(f"  Spearman IC = {ic:+.4f}  p={ic_p:.4f}")


def main():
    pro = init_pro()
    new_share = load_new_share()
    window = load_window()
    print(f"新股样本（已上市）：{len(new_share)} 只")

    features = build_flow_features(new_share, window)
    print(f"成功计算首周资金流特征的新股数：{len(features)}")

    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20160101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))
    index_close = load_index_daily(pro, BENCHMARK_CODE)

    net_df = net_return_analysis(features, trade_days, index_close)
    summarize_overall(net_df)
    summarize_groups(net_df, "week1_avg_amount")
    summarize_groups(net_df, "day1_pct_chg")
    summarize_groups(net_df, "week1_limit_up_days")

    out_dir = pathlib.Path(__file__).parent / "results" / "event_new_share_flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    net_df.to_csv(out_dir / "net_return_summary.csv", index=False)
    features.to_csv(out_dir / "features.csv", index=False)
    print(f"\n结果已保存：{out_dir}")


if __name__ == "__main__":
    main()
