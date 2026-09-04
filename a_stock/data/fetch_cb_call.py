"""
可转债强赎事件数据拉取（cb_call全历史 + cb_basic正股映射）

用于指数增强第十一轮候选③「可转债强制赎回/转股事件对正股冲击」事件研究。
详见 a_stock/docs/research_index_enhancement.md「指数增强策略」第十一轮小节。

数据与方法：
- cb_basic：可转债基本信息（含ts_code到正股stk_code的映射），单次调用
  即返回全量（1162条，不需要分页）
- cb_call：强赎/到赎事件全历史，单次调用固定截断2000条，需offset循环分页
  （已实测：offset=0/2000/3000三段共3062条，第三段62条<2000终止分页）
- 只关心is_call=='公告实施强赎'的记录（真正确认强赎、触发债权人在
  call_reg_date前决定转股或被低价赎回的事件，620+条，2020年后年均50-130次，
  样本量足够统计检验）。'已满足强赎条件'/'公告提示强赎'是前置流程状态，
  '公告不强赎'是公司选择不强赎（无强制转股压力），均不是本方向要测的事件

用法：
  cd a_stock/data
  python fetch_cb_call.py
"""

import pathlib

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "cb_call.parquet"

PAGE_LIMIT = 2000


def fetch_all_cb_call(pro) -> pd.DataFrame:
    chunks = []
    offset = 0
    while True:
        df = pro.cb_call(offset=offset)
        if df.empty:
            break
        chunks.append(df)
        offset += len(df)
        if len(df) < PAGE_LIMIT:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    pro = init_pro()
    call_df = fetch_all_cb_call(pro)
    print(f"cb_call 全历史记录数：{len(call_df)}")

    basic_df = pro.cb_basic()
    print(f"cb_basic 记录数：{len(basic_df)}")

    merged = call_df.merge(
        basic_df[["ts_code", "stk_code", "stk_short_name"]], on="ts_code", how="left"
    )
    for col in ["ann_date", "call_date", "call_reg_date", "payment_date"]:
        merged[col] = pd.to_datetime(merged[col], errors="coerce")

    merged.to_parquet(OUT_PATH, index=False)
    print(f"已保存 {len(merged)} 条记录 -> {OUT_PATH}")

    impl = merged[merged["is_call"] == "公告实施强赎"]
    print(f"'公告实施强赎'事件数：{len(impl)}，独立正股数：{impl['stk_code'].nunique()}")


if __name__ == "__main__":
    main()
