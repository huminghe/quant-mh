"""
全市场ETF跟踪指数（benchmark）拉取，用于"同指数去重候选池"实测

目的：当前机械化候选池（etf_rotation_v23_universe_bias_test.py）把所有流动性达标
ETF当独立候选处理，但A股存在大量"多家基金公司发行同一指数ETF"的情况（如11只
中证A500ETF、9只恒生科技ETF），这些标的收益几乎是同一份的复制品，Top3选股时
同指数多只同时入选并不构成真正的分散，反而可能是集中度风险的隐藏来源。

跟踪指数是ETF的静态属性（成立后基本不变，不是随时间变化的信号），用当前
fund_basic快照获取即不构成前视偏差——这里拉的是"这只ETF跟踪哪个指数"这个事实，
不是任何依赖未来收益/走势推断出的信息。

数据来源：pro.fund_basic(market="E") 的 benchmark 字段（如"沪深300指数收益率×100%"），
清洗后提取核心指数名用于分组。
"""

import pathlib
import re
import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
OUT_PATH = DATA_DIR / "etf_benchmark.parquet"


def clean_benchmark(raw: str) -> str:
    """
    从原始benchmark文本提取核心指数名，去除汇率调整说明、倍数后缀等噪音。
    如"经汇率调整后的中证港股通高股息精选指数收益率×100%" → "中证港股通高股息精选指数"
    """
    if pd.isna(raw):
        return None
    s = str(raw)
    s = re.sub(r"经?汇率调整后?的?", "", s)
    s = re.sub(r"（.*?）|\(.*?\)", "", s)
    s = re.sub(r"收益率.*$", "", s)
    s = re.sub(r"×.*$", "", s)
    s = re.sub(r"指数.*$", "指数", s)
    s = s.strip()
    return s or None


def fetch_etf_benchmark(pro) -> pd.DataFrame:
    """全市场股票型ETF（含已退市）的跟踪指数，字段与 fetch_market_turnover.py 的
    fetch_etf_meta 过滤条件一致，确保代码集合可以直接对齐。"""
    df = pro.fund_basic(market="E", fields="ts_code,name,benchmark,fund_type,status")
    etf = df[df["name"].str.contains("ETF", na=False)]
    etf = etf[etf["fund_type"] == "股票型"]
    etf = etf[etf["status"].isin(["L", "D"])]
    etf = etf[["ts_code", "name", "benchmark"]].reset_index(drop=True)
    etf["benchmark_clean"] = etf["benchmark"].apply(clean_benchmark)
    return etf


def main():
    pro = init_pro()
    print("拉取全市场股票型ETF跟踪指数...")
    df = fetch_etf_benchmark(pro)
    print(f"共 {len(df)} 只，benchmark缺失 {df['benchmark_clean'].isna().sum()} 只")
    df.to_parquet(OUT_PATH, index=False)
    print(f"已保存至 {OUT_PATH}")

    grp = df.dropna(subset=["benchmark_clean"]).groupby("benchmark_clean").size().sort_values(ascending=False)
    print(f"唯一跟踪指数数：{grp.size}（vs ETF总数 {len(df)}）")
    print(f"被2只以上ETF共同跟踪的指数数：{(grp >= 2).sum()}，涉及标的数：{grp[grp >= 2].sum()}")
    print("\n重复度最高的10个指数：")
    print(grp.head(10))


if __name__ == "__main__":
    main()
