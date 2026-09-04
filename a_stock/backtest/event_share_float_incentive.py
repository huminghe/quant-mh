"""
股权激励限售股解禁事件研究（事件驱动信号，仅测核心前提）

背景：指数增强第十一轮候选⑤（a_stock/docs/research_index_enhancement.md「指数增强策略」
第十一轮小节）。候选②（event_share_float_lockup.py）测的是share_float
全部share_type聚合的大额解禁（total_ratio>=5%），事后核查发现该阈值下的
显著结果136/137已经是股权激励限售流通主导——即候选②的假显著/真显著结论
本身已经被股权激励类型解释了大半，重新在同一阈值测同一批事件没有增量
信息。本脚本只筛share_type=="股权激励限售流通"这一类，把阈值降到1%
（该类型聚合后ratio中位数仅0.86%，5%阈值下样本太小），测候选②未覆盖的
中小规模股权激励解禁事件是否有独立信号。

逻辑与候选②一致，两个方向相反的假设都测（不预设结论）：假设A（提前
下跌）——市场提前预期解禁后的减持压力，解禁日前价格已承压；假设B
（解禁后企稳反弹）——压力兑现后不确定性消除，价格企稳反弹。

**注意A股不能做空个股**：假设A即使显著也不能直接做空变现，只能转化为
"回避买入"信号；假设B（解禁后反弹）才是能直接转化为可交易多头信号的
方向。

数据与方法：
- 事件识别：share_float.parquet 筛 share_type=="股权激励限售流通"，按
  ts_code+float_date 聚合，ratio（该类型解禁股数/总股本，百分比）取
  RATIO_THRESHOLD以上。
- 股票池：不限沪深300/中证500，直接用
  share_float_incentive_daily_window.parquet（fetch_share_float_incentive_window.py
  拉取的解禁日前后窗口日线，非复权）。
- 假设A测试（提前下跌）：float_date前PRE_WINDOW_TRADING_DAYS个交易日
  到float_date当天的超额收益，预期为负。
- 假设B测试（解禁后反弹）：float_date后T+1建仓（动态扫描锁死区间，不
  假设次日必然可交易），测20/60个交易日累计超额收益，预期为正。
- 基准：中证800（000906.SH）。

复用 event_index_rebalance.py 的通用工具函数（DRY，与候选②同款）。

用法：
  cd a_stock/backtest
  python event_share_float_incentive.py
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
    shift_trading_day, load_index_daily, index_window_return, ROUND_TRIP_COST,
)

SHARE_FLOAT_FILE = DATA_DIR / "share_float.parquet"
WINDOW_FILE = DATA_DIR / "share_float_incentive_daily_window.parquet"
BENCHMARK_CODE = "000906.SH"  # 中证800

SHARE_TYPE = "股权激励限售流通"
RATIO_THRESHOLD = 1.0  # 解禁比例阈值（百分比），与fetch_share_float_incentive_window.py一致
PRE_WINDOW_TRADING_DAYS = 20  # 假设A：解禁前测试窗口（交易日）
HOLD_WINDOWS = [20, 60]        # 假设B：解禁后持有窗口（交易日）
N_GROUPS = 5
LIMIT_UP_THRESHOLD = 0.095  # 涨停判断阈值（近似，含缓冲，非科创/创业板口径）


def load_events() -> pd.DataFrame:
    if not SHARE_FLOAT_FILE.exists():
        raise FileNotFoundError(f"缺少 {SHARE_FLOAT_FILE}")
    df = pd.read_parquet(SHARE_FLOAT_FILE)
    inc = df[df["share_type"] == SHARE_TYPE]
    event = inc.groupby(["ts_code", "float_date"]).agg(
        ratio=("float_ratio", "sum"), ann_date=("ann_date", "min")
    ).reset_index()
    today = pd.Timestamp.now().normalize()
    clean = event[
        (event["ratio"] >= RATIO_THRESHOLD)
        & (event["ratio"] <= 100)
        & (event["ann_date"] < event["float_date"])
        & (event["float_date"] <= today)
        & (event["float_date"] >= pd.Timestamp("2016-01-01"))
    ].copy()
    return clean


def load_window() -> pd.DataFrame:
    if not WINDOW_FILE.exists():
        raise FileNotFoundError(f"缺少 {WINDOW_FILE}，请先运行 fetch_share_float_incentive_window.py")
    df = pd.read_parquet(WINDOW_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_price_index(window: pd.DataFrame) -> dict:
    idx = {}
    for ts_code, g in window.groupby("ts_code"):
        idx[ts_code] = g.sort_values("trade_date").set_index("trade_date")["close"]
    return idx


def price_on_or_after(series: pd.Series, date: pd.Timestamp) -> float:
    s = series.loc[date:]
    return s.iloc[0] if not s.empty else np.nan


def price_on_or_before(series: pd.Series, date: pd.Timestamp) -> float:
    s = series.loc[:date]
    return s.iloc[-1] if not s.empty else np.nan


def find_first_unlocked_entry(g: pd.DataFrame, start_date: pd.Timestamp) -> tuple:
    after = g[g["trade_date"] > start_date].sort_values("trade_date")
    for _, r in after.iterrows():
        locked = (abs(r["open"] - r["close"]) < 1e-6) and (abs(r["pct_chg"]) / 100.0 >= LIMIT_UP_THRESHOLD)
        if not locked:
            return r["trade_date"], r["open"]
    return None, None


def pre_event_analysis(events: pd.DataFrame, price_idx: dict, trade_days: pd.DatetimeIndex,
                        index_close: pd.Series) -> pd.DataFrame:
    """假设A：解禁前PRE_WINDOW_TRADING_DAYS个交易日到解禁日当天的超额收益"""
    rows = []
    for _, ev in events.iterrows():
        ts_code, float_date = ev["ts_code"], ev["float_date"]
        series = price_idx.get(ts_code)
        if series is None:
            continue
        window_start = shift_trading_day(trade_days, float_date, -PRE_WINDOW_TRADING_DAYS)
        if window_start is None:
            continue
        p_in = price_on_or_after(series, window_start)
        p_out = price_on_or_before(series, float_date)
        if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
            continue
        stock_ret = p_out / p_in - 1
        idx_ret = index_window_return(index_close, window_start, float_date)
        if pd.isna(idx_ret):
            continue
        rows.append({
            "ts_code": ts_code, "float_date": float_date.date(),
            "ratio": ev["ratio"],
            "gross_excess": stock_ret - idx_ret,
        })
    return pd.DataFrame(rows)


def post_event_analysis(events: pd.DataFrame, window: pd.DataFrame, price_idx: dict,
                         trade_days: pd.DatetimeIndex, index_close: pd.Series) -> pd.DataFrame:
    """假设B：解禁后T+1建仓（动态扫描锁死区间），测20/60日累计超额收益"""
    rows = []
    n_all_locked = 0
    for _, ev in events.iterrows():
        ts_code, float_date = ev["ts_code"], ev["float_date"]
        g = window[window["ts_code"] == ts_code]
        if g.empty:
            continue
        entry_date, entry_price_val = find_first_unlocked_entry(g, float_date)
        if entry_date is None:
            n_all_locked += 1
            continue
        series = price_idx.get(ts_code)
        if series is None or pd.isna(entry_price_val) or entry_price_val <= 0:
            continue
        for window_days in HOLD_WINDOWS:
            exit_date = shift_trading_day(trade_days, entry_date, window_days)
            if exit_date is None:
                continue
            p_out = price_on_or_before(series, exit_date)
            if pd.isna(p_out):
                continue
            stock_ret = p_out / entry_price_val - 1
            idx_ret = index_window_return(index_close, entry_date, exit_date)
            if pd.isna(idx_ret):
                continue
            gross_excess = stock_ret - idx_ret
            net_excess = gross_excess - ROUND_TRIP_COST
            rows.append({
                "ts_code": ts_code, "float_date": float_date.date(),
                "entry_date": entry_date.date(), "window": window_days,
                "ratio": ev["ratio"],
                "gross_excess": gross_excess, "net_excess": net_excess,
            })
    print(f"解禁后窗口内始终一字涨跌停锁死（无法参与，已剔除）：{n_all_locked} 只事件")
    return pd.DataFrame(rows)


def summarize(clean: pd.Series, label: str) -> None:
    if clean.empty:
        print(f"{label}：无有效数据")
        return
    mean = clean.mean()
    median = clean.median()
    same_sign = (np.sign(clean) == np.sign(mean)).mean() if mean != 0 else 0.0
    t_stat, p_val = stats.ttest_1samp(clean, 0)
    print(f"{label}：均值={mean:+.4%}  中位数={median:+.4%}  同向占比={same_sign:.1%}  "
          f"n={len(clean)}  t={t_stat:.2f}  p={p_val:.3f}")


def summarize_pre(df: pd.DataFrame) -> None:
    print(f"\n=== 假设A：解禁前{PRE_WINDOW_TRADING_DAYS}个交易日超额收益（预期为负）===")
    summarize(df["gross_excess"].dropna(), "全样本")


def summarize_post(df: pd.DataFrame) -> None:
    print("\n=== 假设B：解禁后T+1建仓净超额收益（扣完整回合成本%.3f%%，预期为正）===" % (ROUND_TRIP_COST * 100))
    for window_days in HOLD_WINDOWS:
        sub = df[df["window"] == window_days]["net_excess"].dropna()
        summarize(sub, f"持有{window_days}个交易日")


def summarize_groups(df: pd.DataFrame) -> None:
    print(f"\n=== 假设B：按ratio（解禁比例）分{N_GROUPS}组（组1=最低，组{N_GROUPS}=最高）===")
    for window_days in HOLD_WINDOWS:
        sub = df[df["window"] == window_days].dropna(subset=["ratio", "net_excess"]).copy()
        if len(sub) < N_GROUPS * 10:
            print(f"\n持有{window_days}个交易日：样本不足，跳过分组")
            continue
        sub["group"] = pd.qcut(sub["ratio"], N_GROUPS, labels=False, duplicates="drop") + 1
        summary = sub.groupby("group")["net_excess"].agg(["mean", "median", "count"])
        print(f"\n持有{window_days}个交易日：")
        print(summary.to_string())
        ic, ic_p = stats.spearmanr(sub["ratio"], sub["net_excess"])
        print(f"  Spearman IC = {ic:+.4f}  p={ic_p:.4f}")


def main():
    pro = init_pro()
    events = load_events()
    print(f"事件数（{SHARE_TYPE}，ratio>={RATIO_THRESHOLD}%）：{len(events)}，"
          f"独立股票数：{events['ts_code'].nunique()}")

    window = load_window()
    price_idx = build_price_index(window)
    print(f"价格窗口数据覆盖股票数：{len(price_idx)}")

    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20160101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))
    index_close = load_index_daily(pro, BENCHMARK_CODE)

    pre_df = pre_event_analysis(events, price_idx, trade_days, index_close)
    summarize_pre(pre_df)

    post_df = post_event_analysis(events, window, price_idx, trade_days, index_close)
    summarize_post(post_df)
    summarize_groups(post_df)

    out_dir = pathlib.Path(__file__).parent / "results" / "event_share_float_incentive"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_df.to_csv(out_dir / "pre_event_summary.csv", index=False)
    post_df.to_csv(out_dir / "post_event_summary.csv", index=False)
    print(f"\n结果已保存：{out_dir}")


if __name__ == "__main__":
    main()
