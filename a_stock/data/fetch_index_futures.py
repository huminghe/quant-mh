"""
股指期货主力连续合约 + 对应现货指数行情数据获取

用于ETF轮动组合股指期货对冲beta中性验证（详见
a_stock/backtest/hedge_beta_test.py）。只拉IF.CFX（沪深300）、
IC.CFX（中证500）两个品种——431池持仓以宽基/行业ETF为主，不含
中证1000相关标的，IM.CFX（2022年才上市，历史覆盖短）暂不拉取。

复用 fetch_futures_commodity.py 的 offset 分页模板（fut_daily单次调用
2000条硬上限）；现货指数用 index_daily，不受截断限制。

用法：
  cd a_stock/data
  python fetch_index_futures.py
"""

import os
import time
import pathlib

import pandas as pd
import tushare as ts

DATA_DIR = pathlib.Path(__file__).parent
OUT_FILE = DATA_DIR / "index_futures_daily.parquet"
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"

START_DATE = "20100101"  # IF.CFX 2010年4月上市，IC.CFX 2015年4月上市
DELAY = 0.35
BATCH_SIZE = 2000

FUTURES = {"IF.CFX": "000300.SH", "IC.CFX": "000905.SH"}
NAMES = {"IF.CFX": "沪深300股指期货", "IC.CFX": "中证500股指期货"}


def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise ValueError("未找到 tushare token（环境变量 TUSHARE_TOKEN 或 ~/.tushare_token）")
    ts.set_token(token)
    return ts.pro_api()


def fetch_main_continuous(pro, ts_code: str) -> pd.DataFrame:
    """拉取主力连续合约全历史（offset分页处理2000条硬上限）"""
    frames = []
    offset = 0
    while True:
        df = pro.fut_daily(
            ts_code=ts_code, start_date=START_DATE, end_date="20261231",
            offset=offset,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount,oi",
        )
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < BATCH_SIZE:
            break
        offset += BATCH_SIZE
        time.sleep(DELAY)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    return result


def fetch_index_daily(pro, ts_code: str) -> pd.DataFrame:
    df = pro.index_daily(ts_code=ts_code, start_date=START_DATE, end_date="20261231")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)[["trade_date", "close"]].rename(
        columns={"close": "idx_close"})


def main():
    pro = init_pro()
    all_rows = []
    for fut_code, idx_code in FUTURES.items():
        print(f"拉取 {fut_code}（{NAMES[fut_code]}）主力连续合约...")
        fut = fetch_main_continuous(pro, fut_code)
        time.sleep(DELAY)
        idx = fetch_index_daily(pro, idx_code)
        merged = fut.merge(idx, on="trade_date", how="inner")
        merged["basis"] = merged["close"] / merged["idx_close"] - 1
        print(f"  {len(merged)} 个交易日，{merged['trade_date'].min().date()} ~ {merged['trade_date'].max().date()}")
        all_rows.append(merged)
        time.sleep(DELAY)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_parquet(OUT_FILE, index=False)
    print(f"\n已保存：{OUT_FILE}（{len(result)} 行）")


if __name__ == "__main__":
    main()
