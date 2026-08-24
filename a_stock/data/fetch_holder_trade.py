"""
股东增减持公告数据拉取（stk_holdertrade，全历史分页拉取）

用于指数增强新候选方向调研：高管/大股东增持公告作为事件驱动信号。详见
a_stock/docs/research.md「指数增强策略」新候选方向调研小节。

已知接口特性（写入前已核查）：
- 单次调用固定截断3000条，长区间必须用offset循环分页拉取，否则静默丢数据
  （已实测：2024全年in_de='IN'不分页只拿到3000条，分页后真实4806条）
- 同一股票短期内常有多条公告（32.8%同ts_code+ann_date有多条记录，
  60.9%相邻公告间隔<=5个交易日），本脚本只做拉取和落盘，事件去重逻辑放在
  下游 event_holder_increase.py（数据获取和事件定义分离，职责单一）

用法：
  cd a_stock/data
  python fetch_holder_trade.py             # 全量拉取 2016-01-01~最新
  python fetch_holder_trade.py --update    # 增量：只拉最近90天
"""

import time
import argparse
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "holder_trade.parquet"

START_DATE = "20160101"
CHUNK_DAYS = 90   # 按90天分段拉取，配合offset分页兜底截断
PAGE_LIMIT = 3000
DELAY = 0.35

FIELDS = ("ts_code,ann_date,holder_name,holder_type,in_de,"
          "change_vol,change_ratio,after_share,avg_price")


def fetch_range(pro, start_date: str, end_date: str) -> pd.DataFrame:
    """分页拉取单个日期区间内的全部记录（offset循环直到不足PAGE_LIMIT为止）"""
    chunks = []
    offset = 0
    while True:
        for attempt in range(3):
            try:
                df = pro.stk_holdertrade(
                    start_date=start_date, end_date=end_date,
                    fields=FIELDS, limit=PAGE_LIMIT, offset=offset,
                )
                break
            except Exception as e:
                print(f"    offset={offset} 第{attempt+1}次失败: {e}")
                time.sleep(1.5)
                df = pd.DataFrame()
        if df is None or df.empty:
            break
        chunks.append(df)
        if len(df) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(DELAY)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def fetch_all(pro, start_date: str, end_date: str) -> pd.DataFrame:
    """按CHUNK_DAYS天分段拉取完整区间，每段内再做offset分页"""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    chunks = []
    cur = start_ts
    n_total_chunks = ((end_ts - start_ts).days // CHUNK_DAYS) + 1
    i = 0
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end_ts)
        s, e = cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        df = fetch_range(pro, s, e)
        if not df.empty:
            chunks.append(df)
        i += 1
        if i % 5 == 0 or cur + pd.Timedelta(days=CHUNK_DAYS) > end_ts:
            print(f"  进度 {i}/{n_total_chunks}（{s}~{e}），累计 {sum(len(c) for c in chunks)} 条")
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(DELAY)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    result["ann_date"] = pd.to_datetime(result["ann_date"])
    result = result.drop_duplicates(
        subset=["ts_code", "ann_date", "holder_name", "in_de", "change_vol"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="股东增减持公告数据拉取（stk_holdertrade）")
    parser.add_argument("--update", action="store_true", help="只拉最近90天（增量更新）")
    args = parser.parse_args()

    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")

    if args.update and OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        start = (pd.Timestamp.today() - pd.Timedelta(days=90)).strftime("%Y%m%d")
        print(f"增量更新：拉取 {start}~{today}")
        new_df = fetch_all(pro, start, today)
        if new_df.empty:
            print("无新数据")
            return
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["ts_code", "ann_date", "holder_name", "in_de", "change_vol"]
        )
    else:
        print(f"全量拉取：{START_DATE}~{today}")
        merged = fetch_all(pro, START_DATE, today)

    if merged.empty:
        print("未获取到任何数据")
        return

    merged = merged.sort_values(["ts_code", "ann_date"]).reset_index(drop=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 {len(merged)} 条记录 -> {OUT_PATH}")
    print(f"覆盖股票数：{merged['ts_code'].nunique()}")
    print(f"公告日范围：{merged['ann_date'].min().date()} ~ {merged['ann_date'].max().date()}")
    print(f"\nin_de 分布：\n{merged['in_de'].value_counts()}")
    print(f"\nholder_type 分布：\n{merged['holder_type'].value_counts()}")


if __name__ == "__main__":
    main()
