"""
可转债正股历史名称变更记录拉取（用于PIT安全的ST状态判断）

用于可转债信用过滤（Phase 2）。详见 a_stock/docs/research_convertible_bond.md。

背景：判断某只正股在历史某个月末是否为ST/*ST，不能用当前最新名称（前视
偏差——退市的ST股可能后来摘帽，也可能反之），必须用 tushare namechange
的历史区间（name + start_date + end_date）做PIT查询：某日期落在某条记录
的[start_date, end_date)区间内，则该日期对应的股票名称就是那条记录的name。

用法：
  cd a_stock/data
  python fetch_cb_stk_namechange.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
UNIVERSE_FILE = DATA_DIR / "cb_universe.parquet"
OUT_PATH = DATA_DIR / "cb_stk_namechange.parquet"

DELAY = 0.15


def fetch_one(pro, ts_code: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return pro.namechange(ts_code=ts_code)
        except Exception as e:
            print(f"    {ts_code} 第{attempt + 1}次失败: {e}")
            time.sleep(1.5)
    return pd.DataFrame()


def main():
    cb = pd.read_parquet(UNIVERSE_FILE)
    stk_codes = cb["stk_code"].dropna().unique().tolist()
    print(f"待拉取正股数：{len(stk_codes)}")

    pro = init_pro()
    chunks = []
    for i, code in enumerate(stk_codes, 1):
        df = fetch_one(pro, code)
        if not df.empty:
            chunks.append(df)
        if i % 200 == 0:
            print(f"  进度 {i}/{len(stk_codes)}")
        time.sleep(DELAY)

    merged = pd.concat(chunks, ignore_index=True)
    for c in ["start_date", "end_date", "ann_date"]:
        merged[c] = pd.to_datetime(merged[c], errors="coerce")
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 {len(merged)} 条记录 -> {OUT_PATH}")
    print(f"覆盖正股数：{merged['ts_code'].nunique()} / {len(stk_codes)}")


if __name__ == "__main__":
    main()
