"""
港股通成分股行业映射获取（同花顺概念/行业指数成分）。

背景：build_etf_sw_exposure.py 用 stock_sw_industry.parquet（申万一级，仅覆盖A股）
给ETF持仓算行业暴露，导致55只港股通主题ETF（如恒生科技/创新药主题ETF）持仓全是
xxxxx.HK代码，匹配率恒为0%，被机械地归入broad而非按其真实主题行业分类
（见 a_stock/docs/research_etf_rotation.md"申万一级行业分层抽样候选池"小节已知局限）。

修复：用 tushare ths_index(exchange="HK", type="I") 获取同花顺港股行业指数
（区别于type="N"概念指数、type="S"特殊名单），再用 ths_member 拉取每个行业的
成分股，构建 港股代码 → 同花顺行业名 的映射，与 stock_sw_industry.parquet
并列使用（两套行业体系不同名，不强行合并到申万一级，下游按数据来源分别处理）。

用法：
  cd a_stock/data
  python fetch_hk_industry.py
"""

import pathlib
import time

import pandas as pd

from fetch_data import get_token
import tushare as ts

DATA_DIR = pathlib.Path(__file__).parent
OUT_FILE = DATA_DIR / "hk_industry.parquet"
DELAY = 0.35


def init_pro():
    ts.set_token(get_token())
    return ts.pro_api()


def fetch_hk_industry(pro) -> pd.DataFrame:
    """拉取同花顺港股行业指数（type=I）及成分股，返回 ts_code(港股代码), hk_industry 两列。"""
    idx = pro.ths_index(exchange="HK")
    idx_i = idx[idx["type"] == "I"]
    print(f"  同花顺港股行业指数：{len(idx_i)} 个")

    rows = []
    for _, row in idx_i.iterrows():
        members = pro.ths_member(ts_code=row["ts_code"])
        for code in members["con_code"]:
            rows.append({"ts_code": code, "hk_industry": row["name"]})
        print(f"    {row['name']}: {len(members)} 只")
        time.sleep(DELAY)

    result = pd.DataFrame(rows).drop_duplicates("ts_code")
    # 统一补齐为5位数字代码，与 fund_portfolio 持仓明细里的 symbol 格式对齐
    num = result["ts_code"].str.replace(".HK", "", regex=False)
    result["ts_code"] = num.str.zfill(5) + ".HK"
    result = result.drop_duplicates("ts_code")
    return result[["ts_code", "hk_industry"]]


def main():
    pro = init_pro()
    df = fetch_hk_industry(pro)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\n完成，共 {len(df)} 只港股，已保存至 {OUT_FILE}")


if __name__ == "__main__":
    main()
