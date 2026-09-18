"""
将 daily/*.parquet 或 stock_daily/*.parquet（日线，已复权）转换为
Qlib dump_bin.py 所需的 CSV 格式：symbol,date,open,close,high,low,volume,vwap,factor

用法：
    python convert_to_qlib.py --src daily --out qlib_csv           # ETF（默认）
    python convert_to_qlib.py --src stock_daily --out qlib_csv_stock  # 个股

输出：a_stock/data/<out>/<symbol>.csv，供 qlib_scripts/dump_bin.py 转为二进制格式。

注意：
- vwap 用 amount/vol 近似（分钟级 vwap 需逐笔数据，此处日线只能近似）；
  tushare amount 单位是千元、vol 单位是手（1手=100份），换算为每份均价需要
  vwap = amount*1000 / (vol*100) = amount*10/vol，不能直接 amount/vol 相除
- factor 固定为 1.0：daily/、stock_daily/ 里的价格已经是复权后的结果
  （ETF 用 fund_adj 手工复权，个股用 tushare pro_bar adj="hfq"），
  不需要 qlib 再乘复权因子
"""
import argparse
import pathlib

import pandas as pd

BASE_DIR = pathlib.Path(__file__).parent


def convert_one(path: pathlib.Path, out_dir: pathlib.Path) -> None:
    df = pd.read_parquet(path)
    symbol = path.stem  # 如 159100.SZ

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "date": df["trade_date"].dt.strftime("%Y-%m-%d"),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["vol"],
            "vwap": df["amount"] * 10 / df["vol"].replace(0, pd.NA),
            "factor": 1.0,
        }
    )
    out["vwap"] = out["vwap"].fillna(out["close"])
    out.to_csv(out_dir / f"{symbol}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="daily", help="源目录名（daily 或 stock_daily）")
    parser.add_argument("--out", default="qlib_csv", help="输出目录名")
    args = parser.parse_args()

    data_dir = BASE_DIR / args.src
    out_dir = BASE_DIR / args.out
    out_dir.mkdir(exist_ok=True)

    files = sorted(data_dir.glob("*.parquet"))
    print(f"共 {len(files)} 个文件待转换")
    for i, f in enumerate(files, 1):
        convert_one(f, out_dir)
        if i % 200 == 0:
            print(f"  已转换 {i}/{len(files)}")
    print(f"完成，输出目录：{out_dir}")


if __name__ == "__main__":
    main()
