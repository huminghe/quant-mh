"""
新股上市后窗口日线数据拉取（daily，非复权，仅取上市后约120个自然日窗口）

用于指数增强第十一轮候选①「新股/次新股上市初期资金流效应」事件研究。详见
a_stock/docs/research.md「指数增强策略」第十一轮小节。

设计说明：
- 只拉上市日起120个自然日的窗口（覆盖上市后约80个交易日，够算首周资金流分组
  变量+60个交易日的后续收益窗口），不拉全历史，避免污染主 stock_daily 目录
  （那里存的是沪深300/中证500历史成分股，新股大多不在这个池子）
- 用 daily（非复权）而非 pro_bar hfq，因为只看上市后短窗口内的相对涨跌，
  除非窗口内发生分红配股（新股上市首年基本不会），不影响信号计算，
  YAGNI——不需要为初筛IC而下载全部复权因子

用法：
  cd a_stock/data
  python fetch_new_share_window.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
NEW_SHARE_FILE = DATA_DIR / "new_share.parquet"
OUT_PATH = DATA_DIR / "new_share_daily_window.parquet"

WINDOW_DAYS = 120  # 上市日起拉取的自然日窗口长度
DELAY = 0.2

FIELDS = "ts_code,trade_date,open,close,pct_chg,vol,amount"


def fetch_one(pro, ts_code: str, issue_date: pd.Timestamp) -> pd.DataFrame:
    start = issue_date.strftime("%Y%m%d")
    end = (issue_date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
    for attempt in range(3):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end, fields=FIELDS)
            return df
        except Exception as e:
            print(f"    {ts_code} 第{attempt + 1}次失败: {e}")
            time.sleep(1.5)
    return pd.DataFrame()


def main():
    if not NEW_SHARE_FILE.exists():
        print(f"缺少 {NEW_SHARE_FILE}，请先运行 fetch_new_share.py")
        return

    new_share = pd.read_parquet(NEW_SHARE_FILE).dropna(subset=["issue_date"])
    pro = init_pro()

    chunks = []
    total = len(new_share)
    for i, (_, row) in enumerate(new_share.iterrows(), 1):
        df = fetch_one(pro, row["ts_code"], row["issue_date"])
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
    print(f"覆盖新股数：{merged['ts_code'].nunique()} / {total}")


if __name__ == "__main__":
    main()
