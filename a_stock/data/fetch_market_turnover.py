"""
全市场ETF历史成交额拉取（用于"标的池选择偏差"实测）

目的：不依赖现有 45 只标的池，重建历史上每个时点"机械化流动性达标"的候选池，
      检验当前标的池的手工圈定是否存在前视偏差（详见 docs/ETF轮动调研.md）。

只拉 amount（成交额），不做复权处理——复权只影响价格走势，不影响成交额本身，
且这一步只用于判断"谁在候选池里"，价格数据留到确定名单后再按需拉取。

数据来源：pro.fund_daily(trade_date=X) 按交易日循环，每次调用返回当天全市场
所有基金的成交数据，比按 ts_code 循环（2000+只基金）效率高得多。
"""

import pathlib
import time
import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "market_turnover.parquet"
UNIVERSE_META_PATH = DATA_DIR / "market_etf_meta.parquet"
START_DATE = "20160101"
END_DATE = pd.Timestamp.today().strftime("%Y%m%d")


def fetch_etf_meta(pro) -> pd.DataFrame:
    """全市场股票型ETF基础信息（含已退市），排除债券/货币/商品ETF"""
    df = pro.fund_basic(market="E")
    etf = df[df["name"].str.contains("ETF", na=False)]
    etf = etf[etf["fund_type"] == "股票型"]
    etf = etf[etf["status"].isin(["L", "D"])]
    return etf[["ts_code", "name", "found_date", "list_date", "delist_date", "status"]].reset_index(drop=True)


def fetch_all_turnover(pro, trade_dates: list, delay: float = 0.35) -> pd.DataFrame:
    """按交易日循环拉取全市场成交额，返回 [trade_date, ts_code, amount] 长表"""
    chunks = []
    n = len(trade_dates)
    for i, d in enumerate(trade_dates):
        for attempt in range(3):
            try:
                df = pro.fund_daily(trade_date=d)
                break
            except Exception as e:
                print(f"  {d} 第{attempt+1}次失败: {e}")
                time.sleep(2)
        else:
            print(f"  {d} 三次重试失败，跳过")
            continue
        if df is not None and not df.empty:
            chunks.append(df[["trade_date", "ts_code", "amount"]])
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{n} ({d})")
        time.sleep(delay)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    pro = init_pro()

    print("拉取ETF基础信息...")
    meta = fetch_etf_meta(pro)
    meta.to_parquet(UNIVERSE_META_PATH, index=False)
    print(f"股票型ETF（含已退市）共 {len(meta)} 只，已保存 {UNIVERSE_META_PATH}")

    print("获取交易日历...")
    cal = pro.trade_cal(exchange="SSE", start_date=START_DATE, end_date=END_DATE, is_open="1")
    trade_dates = sorted(cal["cal_date"].tolist())
    print(f"交易日共 {len(trade_dates)} 个，开始拉取全市场成交额（预计15-20分钟）...")

    turnover = fetch_all_turnover(pro, trade_dates)
    turnover["trade_date"] = pd.to_datetime(turnover["trade_date"])
    turnover.to_parquet(OUT_PATH, index=False)
    print(f"完成，共 {len(turnover)} 行，已保存 {OUT_PATH}")


if __name__ == "__main__":
    main()
