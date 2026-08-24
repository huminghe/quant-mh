"""
可转债强赎事件研究（事件驱动信号，仅测核心前提）

背景：指数增强第十一轮候选③（a_stock/docs/research.md「指数增强策略」
第十一轮小节）。机制：可转债"公告实施强赎"是发行人确认强制赎回，持有人
须在call_reg_date（转股登记日）前决定转股或接受低价（通常接近面值）强制
赎回。触发强赎的前提通常是正股价格已连续多日高于转股价的130%，此时转股
远比接受强赎划算，理性持有人几乎全部选择转股——这会在call_reg_date前后
形成一次性的新增流通股供给（转股新增股份），可能构成短期抛压。

两个方向都测（不预设方向，避免像候选①一样测完才发现无法落地）：
- 假设A（转股期抛压）：ann_date（强赎公告日）到call_reg_date（转股登记日）
  窗口内，新增转股股份逐步进入流通，预期超额收益为负。
- 假设B（压力解除后反弹）：call_reg_date后T+1建仓（动态扫描锁死区间），
  测20/60个交易日累计超额收益，预期为正（抛压兑现后价格企稳/均值回归，
  类似候选②的假设B设计）。

**注意A股不能做空个股**（a_stock/CLAUDE.md）：假设A即使显著也不能直接
做空变现，只能转化为"回避买入"信号；假设B才是能直接转化为可交易多头信号
的方向，这是本方向唯一可能落地的结果。

数据与方法：
- 事件识别：cb_call.parquet中is_call=='公告实施强赎'的记录（唯一应作为
  事件样本的取值，见fetch_cb_call.py说明），630条，独立正股571只。
  call_reg_date缺失的9条用ann_date+35天近似（fetch_cb_call_window.py
  拉取窗口时已用同样近似，此处沿用保持一致）。
- 股票池：不限沪深300/中证500（与stock_daily现有池重叠仅223/571只），
  直接用cb_call_daily_window.parquet（fetch_cb_call_window.py拉取的
  强赎事件前后窗口日线，非复权，理由同候选①②：短窗口不需要复权因子）。
- 测试窗口：先测20个交易日短窗口前提（.claude/lessons.md第99条方法论），
  显著再测60日验证衰减速度。
- 基准：中证800（000906.SH），覆盖沪深300+中证500全部成分股。

复用event_index_rebalance.py的通用工具函数（DRY，同候选②的复用方式）。

用法：
  cd a_stock/backtest
  python event_cb_call.py
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

CB_CALL_FILE = DATA_DIR / "cb_call.parquet"
WINDOW_FILE = DATA_DIR / "cb_call_daily_window.parquet"
BENCHMARK_CODE = "000906.SH"  # 中证800

FALLBACK_GAP_DAYS = 35  # call_reg_date缺失时的近似值，与fetch_cb_call_window.py一致
HOLD_WINDOWS = [20, 60]      # 假设B：转股登记日后持有窗口（交易日）
N_GROUPS = 5
LIMIT_UP_THRESHOLD = 0.095  # 涨停判断阈值（近似，含缓冲，非科创/创业板口径）


def load_events() -> pd.DataFrame:
    if not CB_CALL_FILE.exists():
        raise FileNotFoundError(f"缺少 {CB_CALL_FILE}，请先运行 fetch_cb_call.py")
    df = pd.read_parquet(CB_CALL_FILE)
    events = df[df["is_call"] == "公告实施强赎"].dropna(subset=["stk_code", "ann_date"]).copy()
    events["call_reg_date"] = events["call_reg_date"].fillna(
        events["ann_date"] + pd.Timedelta(days=FALLBACK_GAP_DAYS)
    )
    # cb_call.parquet 自带的 ts_code 是可转债代码，这里只取正股代码 stk_code，
    # 重命名前先只选需要的列，避免与原 ts_code 列重名冲突
    events = events[["stk_code", "ann_date", "call_reg_date"]].rename(columns={"stk_code": "ts_code"})
    return events.drop_duplicates()


def load_window() -> pd.DataFrame:
    if not WINDOW_FILE.exists():
        raise FileNotFoundError(f"缺少 {WINDOW_FILE}，请先运行 fetch_cb_call_window.py")
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
    """从start_date（不含）起扫描，返回第一个未一字涨跌停锁死交易日的
    (trade_date, open价)；若窗口内始终锁死则返回(None, None)。"""
    after = g[g["trade_date"] > start_date].sort_values("trade_date")
    for _, r in after.iterrows():
        locked = (abs(r["open"] - r["close"]) < 1e-6) and (abs(r["pct_chg"]) / 100.0 >= LIMIT_UP_THRESHOLD)
        if not locked:
            return r["trade_date"], r["open"]
    return None, None


def conversion_period_analysis(events: pd.DataFrame, price_idx: dict, trade_days: pd.DatetimeIndex,
                                index_close: pd.Series) -> pd.DataFrame:
    """假设A：ann_date到call_reg_date（转股期）超额收益"""
    rows = []
    for _, ev in events.iterrows():
        ts_code, ann_date, reg_date = ev["ts_code"], ev["ann_date"], ev["call_reg_date"]
        series = price_idx.get(ts_code)
        if series is None:
            continue
        p_in = price_on_or_after(series, ann_date)
        p_out = price_on_or_before(series, reg_date)
        if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
            continue
        stock_ret = p_out / p_in - 1
        idx_ret = index_window_return(index_close, ann_date, reg_date)
        if pd.isna(idx_ret):
            continue
        rows.append({
            "ts_code": ts_code, "ann_date": ann_date.date(), "call_reg_date": reg_date.date(),
            "gross_excess": stock_ret - idx_ret,
        })
    return pd.DataFrame(rows)


def post_event_analysis(events: pd.DataFrame, window: pd.DataFrame, price_idx: dict,
                         trade_days: pd.DatetimeIndex, index_close: pd.Series) -> pd.DataFrame:
    """假设B：转股登记日后T+1建仓（动态扫描锁死区间），测20/60日累计超额收益"""
    rows = []
    n_all_locked = 0
    for _, ev in events.iterrows():
        ts_code, reg_date = ev["ts_code"], ev["call_reg_date"]
        g = window[window["ts_code"] == ts_code]
        if g.empty:
            continue
        entry_date, entry_price_val = find_first_unlocked_entry(g, reg_date)
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
                "ts_code": ts_code, "call_reg_date": reg_date.date(),
                "entry_date": entry_date.date(), "window": window_days,
                "gross_excess": gross_excess, "net_excess": net_excess,
            })
    print(f"转股登记日后窗口内始终一字涨跌停锁死（无法参与，已剔除）：{n_all_locked} 只事件")
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


def summarize_conversion(df: pd.DataFrame) -> None:
    print("\n=== 假设A：转股期（ann_date→call_reg_date）超额收益（预期为负）===")
    summarize(df["gross_excess"].dropna(), "全样本")


def summarize_post(df: pd.DataFrame) -> None:
    print("\n=== 假设B：转股登记日后T+1建仓净超额收益（扣完整回合成本%.3f%%，预期为正）===" % (ROUND_TRIP_COST * 100))
    for window_days in HOLD_WINDOWS:
        sub = df[df["window"] == window_days]["net_excess"].dropna()
        summarize(sub, f"持有{window_days}个交易日")


def main():
    pro = init_pro()
    events = load_events()
    print(f"'公告实施强赎'事件数：{len(events)}，独立正股数：{events['ts_code'].nunique()}")

    window = load_window()
    price_idx = build_price_index(window)
    print(f"价格窗口数据覆盖股票数：{len(price_idx)}")

    trade_days = pd.to_datetime(sorted(pro.trade_cal(
        exchange="SSE", start_date="20040101", end_date="20261231", is_open="1"
    )["cal_date"].tolist()))
    index_close = load_index_daily(pro, BENCHMARK_CODE)

    conv_df = conversion_period_analysis(events, price_idx, trade_days, index_close)
    summarize_conversion(conv_df)

    post_df = post_event_analysis(events, window, price_idx, trade_days, index_close)
    summarize_post(post_df)

    out_dir = pathlib.Path(__file__).parent / "results" / "event_cb_call"
    out_dir.mkdir(parents=True, exist_ok=True)
    conv_df.to_csv(out_dir / "conversion_period_summary.csv", index=False)
    post_df.to_csv(out_dir / "post_event_summary.csv", index=False)
    print(f"\n结果已保存：{out_dir}")


if __name__ == "__main__":
    main()
