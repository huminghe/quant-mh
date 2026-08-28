"""
财报实际披露日期数据获取（disclosure_date）

用于指数增强候选因子"季报披露时点相对排名"（第十七轮候选④）。

假设：管理层倾向延迟披露坏消息、提前披露好消息（学术界"披露时机效应"，
如Earnings Announcement Timing的国际文献），预期同一报告期内更早披露的
公司组，未来收益显著高于更晚披露的公司组。

数据可行性：`disclosure_date`按end_date（报告期）查询单次返回全市场当期
全部记录（实测20231231期5374条，覆盖5374只个股，无需按ts_code循环）。
字段：ts_code/ann_date(最新预约或实际公告日)/end_date(报告期)/
pre_date(最新预约披露日)/actual_date(实际披露日，本候选用这个字段)。

point-in-time：用actual_date（实际披露日）而非end_date（报告期），
避免前视偏差——只有当actual_date到达后，市场才能观察到"这家公司披露早/晚"。

用法：
  cd a_stock/data
  python fetch_disclosure_date.py               # 全量下载（按报告期）
"""

import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "disclosure_date.parquet"
START_YEAR = 2016
FIELDS = "ts_code,ann_date,end_date,pre_date,actual_date"


def get_quarter_end_dates(start_year: int) -> list[str]:
    """全部季报报告期（3/6/9/12月末），到今年最近一个已过去的报告期"""
    today = pd.Timestamp.today()
    dates = []
    for year in range(start_year, today.year + 1):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            d = pd.Timestamp(year=year, month=month, day=day)
            if d <= today:
                dates.append(d.strftime("%Y%m%d"))
    return dates


def run_batch() -> None:
    pro = init_pro()
    end_dates = get_quarter_end_dates(START_YEAR)
    print(f"共 {len(end_dates)} 个报告期，开始拉取披露日期...")

    frames = []
    for i, end_date in enumerate(end_dates, 1):
        df = pro.disclosure_date(end_date=end_date, fields=FIELDS)
        if df is not None and not df.empty:
            frames.append(df)
        if i % 8 == 0 or i <= 3:
            print(f"[{i:02d}/{len(end_dates)}] {end_date} 完成 {len(df) if df is not None else 0} 条")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(OUT_PATH, index=False)
    print(f"\n完成，合计 {len(all_df)} 条，已保存至 {OUT_PATH}")


def load_disclosure_date() -> pd.DataFrame:
    if not OUT_PATH.exists():
        raise FileNotFoundError("disclosure_date.parquet不存在，请先运行本脚本下载")
    df = pd.read_parquet(OUT_PATH)
    for col in ["ann_date", "end_date", "pre_date", "actual_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def main():
    run_batch()


if __name__ == "__main__":
    main()
