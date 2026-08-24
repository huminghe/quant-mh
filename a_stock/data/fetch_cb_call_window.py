"""
可转债强赎事件正股窗口日线数据拉取（daily，非复权，仅取事件日前后窗口）

用于指数增强第十一轮候选③「可转债强制赎回/转股事件对正股冲击」事件研究。
详见 a_stock/docs/research.md「指数增强策略」第十一轮小节。

设计说明：
- 只拉'公告实施强赎'事件对应正股在ann_date前后窗口的日线，不拉全历史，
  避免污染主stock_daily目录（571只正股与该池重叠仅223只）
- 用daily（非复权）而非pro_bar hfq，理由同前两轮新方向脚本：短窗口内
  相对涨跌不需要复权因子，YAGNI
- 窗口跨度：ann_date前WINDOW_DAYS_BEFORE天到call_reg_date后
  WINDOW_DAYS_AFTER天（call_reg_date是最后转股登记日，转股行为在此日
  之前完成，之后是"强赎压力解除"窗口）。call_reg_date缺失的记录（9条）
  退化为用ann_date+35天近似（35天=历史gap_days的90分位附近，覆盖大多数
  情况）

用法：
  cd a_stock/data
  python fetch_cb_call_window.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
CB_CALL_FILE = DATA_DIR / "cb_call.parquet"
OUT_PATH = DATA_DIR / "cb_call_daily_window.parquet"

WINDOW_DAYS_BEFORE = 30
WINDOW_DAYS_AFTER = 75  # 覆盖登记日后60个交易日左右的回测窗口
FALLBACK_GAP_DAYS = 35  # call_reg_date缺失时，用ann_date+此值近似
DELAY = 0.2

FIELDS = "ts_code,trade_date,open,close,pct_chg,vol,amount"


def load_events() -> pd.DataFrame:
    if not CB_CALL_FILE.exists():
        raise FileNotFoundError(f"缺少 {CB_CALL_FILE}，请先运行 fetch_cb_call.py")
    df = pd.read_parquet(CB_CALL_FILE)
    impl = df[df["is_call"] == "公告实施强赎"].dropna(subset=["stk_code", "ann_date"]).copy()
    impl["call_reg_date"] = impl["call_reg_date"].fillna(
        impl["ann_date"] + pd.Timedelta(days=FALLBACK_GAP_DAYS)
    )
    return impl


def fetch_one(pro, stk_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    for attempt in range(3):
        try:
            return pro.daily(ts_code=stk_code, start_date=start_s, end_date=end_s, fields=FIELDS)
        except Exception as e:
            print(f"    {stk_code} 第{attempt + 1}次失败: {e}")
            time.sleep(1.5)
    return pd.DataFrame()


def merge_spans(spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """合并同一正股多次强赎事件的窗口区间（重叠/相邻则合并），减少接口调用"""
    raw = sorted(spans)
    merged = [raw[0]]
    for s, e in raw[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + pd.Timedelta(days=1):
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def main():
    events = load_events()
    print(f"事件数：{len(events)}，独立正股数：{events['stk_code'].nunique()}")

    events["window_start"] = events["ann_date"] - pd.Timedelta(days=WINDOW_DAYS_BEFORE)
    events["window_end"] = events["call_reg_date"] + pd.Timedelta(days=WINDOW_DAYS_AFTER)

    by_stock = events.groupby("stk_code").apply(
        lambda g: list(zip(g["window_start"], g["window_end"])), include_groups=False
    )
    spans = [(code, s, e) for code, sp in by_stock.items() for s, e in merge_spans(sp)]
    print(f"合并后窗口区间数：{len(spans)}")

    pro = init_pro()
    chunks = []
    total = len(spans)
    for i, (stk_code, start, end) in enumerate(spans, 1):
        df = fetch_one(pro, stk_code, start, end)
        if not df.empty:
            chunks.append(df)
        if i % 100 == 0:
            print(f"  进度 {i}/{total}，累计 {sum(len(c) for c in chunks)} 条")
        time.sleep(DELAY)

    if not chunks:
        print("未获取到任何数据")
        return

    merged = pd.concat(chunks, ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"])
    merged = merged.drop_duplicates(subset=["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 {len(merged)} 条记录 -> {OUT_PATH}")
    print(f"覆盖正股数：{merged['ts_code'].nunique()} / {events['stk_code'].nunique()}")


if __name__ == "__main__":
    main()
