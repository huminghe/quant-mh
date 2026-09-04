"""
新股发行/上市清单拉取（new_share）

用于指数增强第十一轮候选①「新股/次新股上市初期资金流效应」调研。详见
a_stock/docs/research_index_enhancement.md「指数增强策略」第十一轮小节。

已知接口特性（写入前已核查）：
- 单次调用固定截断2000条，需offset循环分页（已实测：全历史合计约3016条，
  不分页只拿最近2000条）
- issue_date（上市日）少量记录为空（约13条，主要是尚未上市或退市代码复用的
  历史遗留情况），下游脚本需自行过滤

用法：
  cd a_stock/data
  python fetch_new_share.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "new_share.parquet"

START_DATE = "20160101"
PAGE_LIMIT = 2000
DELAY = 0.35

FIELDS = "ts_code,name,ipo_date,issue_date,amount,price,pe,funds"


def fetch_all(pro, start_date: str, end_date: str) -> pd.DataFrame:
    chunks = []
    offset = 0
    while True:
        for attempt in range(3):
            try:
                df = pro.new_share(
                    start_date=start_date, end_date=end_date,
                    fields=FIELDS, limit=PAGE_LIMIT, offset=offset,
                )
                break
            except Exception as e:
                print(f"  offset={offset} 第{attempt + 1}次失败: {e}")
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
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["ts_code"])


def main():
    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")
    print(f"拉取新股清单：{START_DATE}~{today}")
    df = fetch_all(pro, START_DATE, today)
    if df.empty:
        print("未获取到任何数据")
        return

    df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.sort_values("ipo_date").reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\n已保存 {len(df)} 条记录 -> {OUT_PATH}")
    n_missing = df["issue_date"].isna().sum()
    print(f"issue_date缺失（尚未上市/历史遗留）：{n_missing}")
    listed = df.dropna(subset=["issue_date"])
    print(f"已上市：{len(listed)}，上市日范围：{listed['issue_date'].min().date()} ~ {listed['issue_date'].max().date()}")


if __name__ == "__main__":
    main()
