"""
全A股月度估值快照拉取（PE_TTM/PB），用于ETF轮动问题②估值类行业信号IC检验

背景：行业景气度基本面（净利润增速/ROE/现金流质量/杠杆/营收增速）5个指标
已全部测试排除（详见etf_rotation_v37/v39）。这些都是"成长/景气度"类信号，
估值(PE/PB)是完全不同的信号逻辑——不是"在变好吗"，是"贵不贵"，均值回归而非
趋势跟踪。此前从未在ETF轮动IC框架里测过，值得单独测。

只拉月度调仓日（每月首个交易日，全历史仅128个，非全部约2700个交易日），
用tushare daily_basic接口按trade_date循环（一次调用返回当天全市场，比按
ts_code循环快得多），避免不必要的数据量（YAGNI——不需要日频估值，只需要
月度截面）。
"""

import pathlib
import time

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "valuation_monthly.parquet"
START_DATE = "20160101"


def get_monthly_rebal_dates(pro) -> list:
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=START_DATE, end_date=end_date, is_open="1")
    dates = pd.to_datetime(sorted(cal["cal_date"].tolist()))
    df = pd.DataFrame(index=dates)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


def fetch_monthly_valuation(pro, dates: list, delay: float = 0.35) -> pd.DataFrame:
    chunks = []
    n = len(dates)
    for i, d in enumerate(dates):
        trade_date = d.strftime("%Y%m%d")
        for attempt in range(3):
            try:
                df = pro.daily_basic(trade_date=trade_date, fields="ts_code,trade_date,pe_ttm,pb")
                break
            except Exception as e:
                print(f"  {trade_date} 第{attempt+1}次失败: {e}")
                time.sleep(2)
        else:
            print(f"  {trade_date} 三次重试失败，跳过")
            continue
        if df is not None and not df.empty:
            chunks.append(df)
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{n} ({trade_date})")
        time.sleep(delay)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    pro = init_pro()
    print("获取月度调仓日历...")
    dates = get_monthly_rebal_dates(pro)
    print(f"共 {len(dates)} 个月度评估点，{dates[0].date()} ~ {dates[-1].date()}")

    valuation = fetch_monthly_valuation(pro, dates)
    valuation["trade_date"] = pd.to_datetime(valuation["trade_date"])
    valuation.to_parquet(OUT_PATH, index=False)
    print(f"完成，共 {len(valuation)} 行，已保存 {OUT_PATH}")
    print(f"pe_ttm缺失率={valuation['pe_ttm'].isna().mean():.1%}，pb缺失率={valuation['pb'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
