"""
申万一级行业成分股映射获取

- 用 tushare index_classify（申万2021分类）获取行业代码列表
- 用 index_member_all 获取每个行业下的历史成分股
- 缓存为 Parquet：ts_code → 申万一级行业名

注意：index_member_all 返回的 out_date 全部为空（当前账号权限下），
即该映射是"当前"成分股快照，非严格 point-in-time。
用于回测存在轻微前视偏差，但优于完全无行业映射。

用法：
  cd a_stock/data
  python fetch_sw_industry.py
"""

import os
import pathlib
import time

import pandas as pd
import tushare as ts

DATA_DIR   = pathlib.Path(__file__).parent
OUT_FILE   = DATA_DIR / "stock_sw_industry.parquet"
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"
DELAY      = 0.3


def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise ValueError("未找到 tushare token（环境变量 TUSHARE_TOKEN 或 ~/.tushare_token）")
    ts.set_token(token)
    return ts.pro_api()


def fetch_sw_industry(pro) -> pd.DataFrame:
    """拉取申万一级行业分类及各行业成分股，返回 ts_code, sw_industry 两列"""
    cls = pro.index_classify(level="L1", src="SW2021")
    print(f"  申万一级行业：{len(cls)} 个")

    rows = []
    for _, row in cls.iterrows():
        l1_code = row["index_code"]
        l1_name = row["industry_name"]
        df = pro.index_member_all(l1_code=l1_code)
        for ts_code in df["ts_code"].unique():
            rows.append({"ts_code": ts_code, "sw_industry": l1_name})
        print(f"    {l1_name}: {len(df)} 只")
        time.sleep(DELAY)

    result = pd.DataFrame(rows).drop_duplicates("ts_code")
    return result


def main():
    pro = init_pro()
    df = fetch_sw_industry(pro)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\n完成，共 {len(df)} 只股票，已保存至 {OUT_FILE}")


if __name__ == "__main__":
    main()
