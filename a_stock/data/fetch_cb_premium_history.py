"""
全市场可转债历史双低时间序列拉取（akshare bond_zh_cov_value_analysis）

用于可转债双低策略验证。详见 a_stock/docs/research_convertible_bond.md。

背景：
- tushare cb_price_chg 接口无访问权限，cb_basic.conv_price 是静态值（忽略
  下修事件），都不能提供逐日转股溢价率历史序列
- akshare bond_zh_cov_value_analysis(symbol) 提供逐日「收盘价/纯债价值/
  转股价值/纯债溢价率/转股溢价率」完整历史序列，已核实覆盖已退市标的
  （无幸存者偏差），小批量测试无明显限流（约0.23秒/只）
- symbol 参数用不含交易所后缀的纯数字代码（如 '128114'，对应tushare的
  '128114.SZ'），与cb_daily的ts_code不同，需要转换

用法：
  cd a_stock/data
  python fetch_cb_premium_history.py
"""

import time
import pathlib

import pandas as pd
import akshare as ak

DATA_DIR = pathlib.Path(__file__).parent
UNIVERSE_PATH = DATA_DIR / "cb_universe.parquet"
OUT_PATH = DATA_DIR / "cb_premium_history.parquet"

DELAY = 0.1
RETRY = 3


def ts_to_ak_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


def fetch_one(symbol: str) -> pd.DataFrame:
    for attempt in range(RETRY):
        try:
            df = ak.bond_zh_cov_value_analysis(symbol=symbol)
            return df
        except Exception as e:
            print(f"    {symbol} 第{attempt + 1}次失败: {e}")
            time.sleep(1.0)
    return pd.DataFrame()


def main():
    cb = pd.read_parquet(UNIVERSE_PATH)
    codes = cb["ts_code"].tolist()
    print(f"待拉取转债数：{len(codes)}")

    chunks = []
    failed = []
    for i, ts_code in enumerate(codes, 1):
        symbol = ts_to_ak_symbol(ts_code)
        df = fetch_one(symbol)
        if df.empty:
            failed.append(ts_code)
        else:
            df = df.rename(columns={
                "日期": "trade_date", "收盘价": "close",
                "纯债价值": "straight_value", "转股价值": "conv_value",
                "纯债溢价率": "straight_premium_pct", "转股溢价率": "conv_premium_pct",
            })
            df["ts_code"] = ts_code
            chunks.append(df)
        if i % 100 == 0:
            print(f"  进度 {i}/{len(codes)}，累计成功 {len(chunks)}，失败 {len(failed)}")
        time.sleep(DELAY)

    if not chunks:
        print("未获取到任何数据")
        return

    merged = pd.concat(chunks, ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"])
    merged = merged.drop_duplicates(subset=["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 {len(merged)} 条记录 -> {OUT_PATH}")
    print(f"覆盖转债数：{merged['ts_code'].nunique()} / {len(codes)}")
    if failed:
        print(f"失败 {len(failed)} 只：{failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
