"""
业绩预告数据拉取（forecast_vip，按period批量拉取全市场）

用于指数增强新候选因子调研：业绩预告类型（预增/预减/略增/略减/续盈/续亏/
首亏/扭亏/不确定）作为盈利预期修正信号的粗粒度替代（分析师预测明细接口
report_rc限流10次/天不可用）。详见 a_stock/docs/research_index_enhancement.md「指数增强
策略」新候选因子调研小节。

已知局限（写入前已核查，供后续IC验证脚本引用）：A股业绩预告只在业绩大幅
变动时强制披露，是选择性事件，覆盖率远低于EPS/PE等普遍存在的字段。抽样
检查沪深300+中证500成分股（1574只）：季报覆盖率7-35%，年报覆盖率约
48-50%，均低于项目60%覆盖率门限。IC验证仅供参考，不作为正式候选。

用法：
  cd a_stock/data
  python fetch_forecast.py             # 全量拉取 2016Q1~最新季度
  python fetch_forecast.py --update    # 增量：只拉最近4个季度
"""

import time
import argparse
import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "forecast.parquet"

START_YEAR = 2016
DELAY = 0.35

FIELDS = ("ts_code,ann_date,end_date,type,p_change_min,p_change_max,"
          "net_profit_min,net_profit_max,last_parent_net")


def get_periods(start_year: int, end_year: int) -> list[str]:
    periods = []
    for y in range(start_year, end_year + 1):
        for q_end in ("0331", "0630", "0930", "1231"):
            periods.append(f"{y}{q_end}")
    today = pd.Timestamp.today().strftime("%Y%m%d")
    return [p for p in periods if p <= today]


def fetch_all_periods(pro, periods: list[str]) -> pd.DataFrame:
    chunks = []
    n = len(periods)
    for i, period in enumerate(periods):
        for attempt in range(3):
            try:
                df = pro.forecast_vip(period=period, fields=FIELDS)
                break
            except Exception as e:
                print(f"  {period} 第{attempt+1}次失败: {e}")
                time.sleep(1.5)
                df = pd.DataFrame()
        if not df.empty:
            chunks.append(df)
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"  进度 {i+1}/{n}（{period}），累计 {sum(len(c) for c in chunks)} 条")
        time.sleep(DELAY)
    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    result["ann_date"] = pd.to_datetime(result["ann_date"])
    result["end_date"] = pd.to_datetime(result["end_date"])
    result = result.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
    return result


def main():
    parser = argparse.ArgumentParser(description="业绩预告数据拉取（forecast_vip）")
    parser.add_argument("--update", action="store_true", help="只拉最近4个季度（增量更新）")
    args = parser.parse_args()

    pro = init_pro()
    end_year = pd.Timestamp.today().year

    if args.update and OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        recent_year = end_year - 1
        periods = get_periods(recent_year, end_year)
        print(f"增量更新：拉取 {periods[0]}~{periods[-1]}（{len(periods)}个period）")
        new_df = fetch_all_periods(pro, periods)
        if new_df.empty:
            print("无新数据")
            return
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
    else:
        periods = get_periods(START_YEAR, end_year)
        print(f"全量拉取：{periods[0]}~{periods[-1]}（{len(periods)}个period）")
        merged = fetch_all_periods(pro, periods)

    if merged.empty:
        print("未获取到任何数据")
        return

    merged = merged.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
    merged.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 {len(merged)} 条记录 -> {OUT_PATH}")
    print(f"覆盖股票数：{merged['ts_code'].nunique()}")
    print(f"报告期范围：{merged['end_date'].min().date()} ~ {merged['end_date'].max().date()}")
    print(f"\ntype 分布：\n{merged['type'].value_counts()}")


if __name__ == "__main__":
    main()
