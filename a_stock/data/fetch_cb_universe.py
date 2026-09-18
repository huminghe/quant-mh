"""
全市场可转债基本信息 + 日线（含已退市）+ 正股映射拉取

用于可转债双低策略验证。详见 a_stock/docs/research_convertible_bond.md。

数据与方法：
- cb_basic：全量可转债基本信息（1165条，含cb_type非'CB'的可交换债EB等，
  过滤后CB类型1138条），单次调用返回全量，含delist_date（已退市也保留）
- cb_daily：逐只转债日线，按ts_code循环拉取全历史（不指定日期范围时
  tushare已验证无截断，980条/只测试通过），含已退市标的的历史数据
  （非停牌期无数据，停牌/未上市期收盘价为0，不做特殊处理，下游按
  vol>0过滤即可）

用法：
  cd a_stock/data
  python fetch_cb_universe.py
"""

import time
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
UNIVERSE_PATH = DATA_DIR / "cb_universe.parquet"
DAILY_PATH = DATA_DIR / "cb_daily_all.parquet"

DELAY = 0.15
DATE_COLS = ["value_date", "maturity_date", "list_date", "delist_date",
             "conv_start_date", "conv_end_date", "conv_stop_date"]


def fetch_universe(pro) -> pd.DataFrame:
    df = pro.cb_basic()
    cb = df[df["cb_type"] == "CB"].copy()
    for c in DATE_COLS:
        cb[c] = pd.to_datetime(cb[c], errors="coerce")
    return cb


def fetch_daily_all(pro, codes: list[str]) -> pd.DataFrame:
    chunks = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        for attempt in range(3):
            try:
                d = pro.cb_daily(ts_code=code)
                break
            except Exception as e:
                print(f"    {code} 第{attempt + 1}次失败: {e}")
                time.sleep(1.5)
                d = pd.DataFrame()
        if not d.empty:
            chunks.append(d)
        if i % 100 == 0:
            print(f"  进度 {i}/{total}，累计 {sum(len(c) for c in chunks)} 条")
        time.sleep(DELAY)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    pro = init_pro()

    cb = fetch_universe(pro)
    print(f"CB类型转债数：{len(cb)}，已退市：{cb['delist_date'].notna().sum()}，存续：{cb['delist_date'].isna().sum()}")
    cb.to_parquet(UNIVERSE_PATH, index=False)
    print(f"已保存 -> {UNIVERSE_PATH}")

    codes = cb["ts_code"].tolist()
    daily = fetch_daily_all(pro, codes)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.drop_duplicates(subset=["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])
    daily.to_parquet(DAILY_PATH, index=False)
    print(f"\n已保存 {len(daily)} 条日线记录 -> {DAILY_PATH}")
    print(f"覆盖转债数：{daily['ts_code'].nunique()} / {len(codes)}")


if __name__ == "__main__":
    main()
