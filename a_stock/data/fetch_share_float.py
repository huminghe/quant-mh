"""
限售解禁数据拉取（用于指数增强另类数据因子：即将解禁比例）

背景：限售解禁临近时，市场存在减持预期，可能压制股价——这是此前从未
测试过的另类数据信号，与已证伪的基本面景气度类因子（净利润增速/ROE/
现金流质量/杠杆/营收增速）逻辑完全不同。

用tushare share_float接口，按公告日期(ann_date)分页拉取全市场解禁记录
（该接口不限制ts_code循环，直接按日期范围批量拉取更省调用次数）。
单次调用上限6000行，用offset分页拉全。

字段：float_share(解禁数量，万股)，float_ratio(占总股本比例，已算好，
不需要额外用市值折算)。

用法：
  cd a_stock/data
  python fetch_share_float.py
"""

import os
import time
import pathlib
import pandas as pd
import tushare as ts

DATA_DIR   = pathlib.Path(__file__).parent
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"
OUT_PATH   = DATA_DIR / "share_float.parquet"

START_DATE = "20160101"
END_DATE   = "20260801"

PAGE_SIZE = 6000
DELAY = 0.3


def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise ValueError("未找到 tushare token，请设置环境变量 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


def fetch_range_flat(pro, start: str, end: str, label: str) -> tuple[list[pd.DataFrame], bool]:
    """按offset单次拉取一个日期区间，不做二分。返回(结果列表, 是否遇到截断)"""
    dfs = []
    offset = 0
    while True:
        try:
            df = pro.share_float(start_date=start, end_date=end, offset=offset)
        except Exception as e:
            print(f"    {label} offset={offset}: 失败 {e}，重试一次...")
            time.sleep(1.0)
            try:
                df = pro.share_float(start_date=start, end_date=end, offset=offset)
            except Exception as e2:
                print(f"    {label} offset={offset}: 重试仍失败 {e2}，该区间需二分")
                return dfs, True
        if df.empty:
            break
        dfs.append(df)
        print(f"    {label} offset={offset}: 获取 {len(df)} 行")
        if len(df) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(DELAY)
    return dfs, False


def fetch_range(pro, start: str, end: str, label: str, depth: int = 0) -> list[pd.DataFrame]:
    """按offset分页拉取，遇到截断时二分日期区间递归重试，而非放弃剩余分页"""
    dfs, truncated = fetch_range_flat(pro, start, end, label)
    if not truncated:
        return dfs
    if depth >= 6:
        print(f"  {label} 二分深度已达上限({depth})，仍失败，放弃该区间剩余部分")
        return dfs
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    if start_dt >= end_dt:
        print(f"  {label} 区间已不可再分，放弃")
        return dfs
    mid_dt = start_dt + (end_dt - start_dt) / 2
    mid_start = mid_dt.strftime("%Y%m%d")
    mid_end_prev = (mid_dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
    print(f"  {label} 触发截断，二分为 [{start}, {mid_end_prev}] + [{mid_start}, {end}]")
    left = fetch_range(pro, start, mid_end_prev, f"{label}-左", depth + 1)
    right = fetch_range(pro, mid_start, end, f"{label}-右", depth + 1)
    return left + right


def fetch_all(pro) -> pd.DataFrame:
    """按季度拉取（明细颗粒度到股东，年度数据量可能超过offset上限，按季度更稳；
    单季度仍触发截断时 fetch_range 会自动二分日期区间重试）"""
    all_dfs = []
    for year in range(2016, 2027):
        for q_start, q_end in [("0101", "0331"), ("0401", "0630"),
                               ("0701", "0930"), ("1001", "1231")]:
            s = f"{year}{q_start}"
            e = f"{year}{q_end}"
            label = f"{year}Q{['0101','0401','0701','1001'].index(q_start)+1}"
            dfs = fetch_range(pro, s, e, label)
            all_dfs.extend(dfs)
            time.sleep(DELAY)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates()


def main():
    pro = init_pro()
    print("拉取限售解禁数据（2016-2026，按季度分页，遇截断自动二分重试）...")
    df = fetch_all(pro)
    if df.empty:
        print("未获取到数据")
        return

    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df["float_date"] = pd.to_datetime(df["float_date"], format="%Y%m%d", errors="coerce")
    df["float_ratio"] = pd.to_numeric(df["float_ratio"], errors="coerce")
    df["float_share"] = pd.to_numeric(df["float_share"], errors="coerce")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n完成，共 {len(df)} 行，已保存 {OUT_PATH}")
    print(f"覆盖股票数: {df['ts_code'].nunique()}")
    print(f"float_date范围: {df['float_date'].min()} ~ {df['float_date'].max()}")
    print(f"float_ratio缺失率: {df['float_ratio'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
