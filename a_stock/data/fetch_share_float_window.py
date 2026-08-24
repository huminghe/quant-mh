"""
限售股解禁事件窗口日线数据拉取（daily，非复权，仅取解禁日前后约90自然日窗口）

用于指数增强第十一轮候选②「限售股解禁强制平仓风险」事件研究。详见
a_stock/docs/research.md「指数增强策略」第十一轮小节。

设计说明：
- 只拉解禁日前后窗口（不拉全历史），避免污染主 stock_daily 目录（那里存的是
  沪深300/中证500历史成分股，解禁事件覆盖的5277只股票与之重叠仅1321只）
- 用 daily（非复权）而非 pro_bar hfq，理由同 fetch_new_share_window.py：
  短窗口内相对涨跌不需要复权因子，YAGNI
- 只拉 total_ratio>=5% 的大额解禁事件对应股票（对应 share_float.parquet 里
  按 ts_code+float_date 聚合后的解禁比例），小额解禁抛压意义不大，不浪费
  接口调用额度

用法：
  cd a_stock/data
  python fetch_share_float_window.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
SHARE_FLOAT_FILE = DATA_DIR / "share_float.parquet"
OUT_PATH = DATA_DIR / "share_float_daily_window.parquet"

RATIO_THRESHOLD = 5.0  # 解禁比例阈值（total_ratio，百分比），低于此不拉取
WINDOW_DAYS_BEFORE = 60  # 解禁日前窗口（自然日），覆盖约40个交易日
WINDOW_DAYS_AFTER = 45  # 解禁日后窗口（自然日），覆盖约30个交易日
DELAY = 0.2

FIELDS = "ts_code,trade_date,open,close,pct_chg,vol,amount"


def build_event_list() -> pd.DataFrame:
    df = pd.read_parquet(SHARE_FLOAT_FILE)
    event = df.groupby(["ts_code", "float_date"]).agg(
        total_ratio=("float_ratio", "sum"), ann_date=("ann_date", "min")
    ).reset_index()
    today = pd.Timestamp.now().normalize()
    clean = event[
        (event["total_ratio"] >= RATIO_THRESHOLD)
        & (event["total_ratio"] <= 100)
        & (event["ann_date"] < event["float_date"])
        & (event["float_date"] <= today)
        & (event["float_date"] >= pd.Timestamp("2016-01-01"))
    ].copy()
    return clean


def merge_spans(dates: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """把同一股票的多次解禁事件窗口([float_date-BEFORE, float_date+AFTER])合并成
    不重叠/不相邻的区间列表，减少接口调用次数（一只股票多次解禁事件时间相近很常见）"""
    raw = sorted(
        (d - pd.Timedelta(days=WINDOW_DAYS_BEFORE), d + pd.Timedelta(days=WINDOW_DAYS_AFTER))
        for d in dates
    )
    merged = [raw[0]]
    for s, e in raw[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + pd.Timedelta(days=1):
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def fetch_one(pro, ts_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    for attempt in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_s, end_date=end_s, fields=FIELDS)
            return df
        except Exception as e:
            print(f"    {ts_code} 第{attempt + 1}次失败: {e}")
            time.sleep(1.5)
    return pd.DataFrame()


def main():
    if not SHARE_FLOAT_FILE.exists():
        print(f"缺少 {SHARE_FLOAT_FILE}")
        return

    events = build_event_list()
    codes = sorted(events["ts_code"].unique())
    print(f"事件数：{len(events)}，独立股票数：{len(codes)}")

    by_code = events.groupby("ts_code")["float_date"].apply(list)
    spans = [(code, s, e) for code, dates in by_code.items() for s, e in merge_spans(dates)]
    print(f"合并后窗口区间数：{len(spans)}（合并前事件数：{len(events)}）")

    pro = init_pro()
    chunks = []
    total = len(spans)
    for i, (ts_code, start, end) in enumerate(spans, 1):
        df = fetch_one(pro, ts_code, start, end)
        if not df.empty:
            chunks.append(df)
        if i % 200 == 0:
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
    print(f"覆盖股票数：{merged['ts_code'].nunique()} / {len(codes)}")


if __name__ == "__main__":
    main()
