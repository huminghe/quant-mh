"""
个股融券余额数据获取（margin_detail，字段rqye）

用于指数增强候选因子"融券余额环比变化"（第十七轮候选③）。

与第十三轮已证伪的margin_balance因子区分：margin_balance用的是`rzrqye`
（融资融券合计）且按行业聚合；本候选专用`rqye`（融券余额，单独字段，
不含融资部分）、个股层面，是全新角度，不是重复第十三轮的构造方式。

数据可行性核查（已实测确认）：margin_detail按trade_date查询单日全市场
返回，对沪深300+中证500历史成分股并集（1578只）覆盖率88.1%（2024-03-29
样本），远超60%可行性红线。2016年之前覆盖率明显更低（2016-01-04约913条，
2010年只有41条），故起始日期定为2016-01-01，与项目其他因子研究窗口一致。

用法：
  cd a_stock/data
  python fetch_short_balance.py               # 全量下载（按月末交易日）
"""

import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
CACHE_DIR = DATA_DIR / "short_balance_cache"
START_DATE = "20160101"

EMPTY_SCHEMA = ["trade_date", "ts_code", "rqye"]


def get_month_end_trading_days(pro) -> list[pd.Timestamp]:
    """同fetch_index_members.py的月末交易日修正逻辑：不能直接用自然日历
    月末去查询，当月最后一天若非交易日会导致数据缺失。"""
    today = pd.Timestamp.today()
    cal = pro.trade_cal(exchange="SSE", start_date=START_DATE,
                         end_date=today.strftime("%Y%m%d"), is_open="1")
    trade_days = pd.to_datetime(sorted(cal["cal_date"]))
    trade_days_s = pd.Series(trade_days)
    months = (trade_days_s
              .groupby(trade_days_s.dt.to_period("M"))
              .max()
              .tolist())
    return months


def run_batch() -> None:
    pro = init_pro()
    CACHE_DIR.mkdir(exist_ok=True)
    months = get_month_end_trading_days(pro)
    print(f"共 {len(months)} 个月末交易日，开始拉取融券余额...")

    for i, month_end in enumerate(months, 1):
        date_str = month_end.strftime("%Y%m%d")
        path = CACHE_DIR / f"{date_str}.parquet"
        if path.exists():
            continue
        df = pro.margin_detail(trade_date=date_str, fields="trade_date,ts_code,rqye")
        if df is None or df.empty:
            df = pd.DataFrame(columns=EMPTY_SCHEMA)
        df.to_parquet(path, index=False)
        if i % 20 == 0 or i <= 3:
            print(f"[{i:03d}/{len(months)}] {date_str} 完成 {len(df)} 条")

    print("完成。")


def load_short_balance_panel() -> pd.DataFrame:
    """读取全部月末快照，返回宽格式面板：index=trade_date，columns=ts_code，值=rqye。"""
    frames = []
    for path in sorted(CACHE_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("没有找到任何融券余额数据，请先运行本脚本下载")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["trade_date"] = pd.to_datetime(all_df["trade_date"])
    panel = all_df.pivot_table(index="trade_date", columns="ts_code", values="rqye", aggfunc="last")
    return panel.sort_index()


def main():
    run_batch()


if __name__ == "__main__":
    main()
