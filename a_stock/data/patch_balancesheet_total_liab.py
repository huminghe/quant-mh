"""
给已缓存的 balancesheet/{ts_code}.parquet 补充 total_liab（总负债）字段。

背景：factor_ic_quality_v2.py 首次拉取资产负债表时只要了
total_assets/total_cur_assets/money_cap/total_cur_liab/st_borr，
没有 total_liab，导致 Ndp（净负债/价格）因子无法计算。
这里对已有缓存文件做增量字段合并，不重新拉取已有字段，只追加 total_liab。

用法：
  cd a_stock/data
  python patch_balancesheet_total_liab.py
"""

import os
import time
import pathlib

import pandas as pd
import tushare as ts

DATA_DIR = pathlib.Path(__file__).parent
BS_DIR = DATA_DIR / "balancesheet"
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"

DELAY = 0.35


def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    ts.set_token(token)
    return ts.pro_api()


def main():
    pro = init_pro()
    files = sorted(BS_DIR.glob("*.parquet"))
    print(f"待处理：{len(files)} 只")

    ok, skip, fail = 0, 0, 0
    for i, path in enumerate(files, 1):
        ts_code = path.stem
        df = pd.read_parquet(path)

        if "total_liab" in df.columns:
            skip += 1
            continue

        try:
            liab = pro.balancesheet(
                ts_code=ts_code, start_date="20130101",
                fields="ts_code,end_date,total_liab",
            )
            if liab is None or liab.empty:
                fail += 1
                continue
            liab["end_date"] = pd.to_datetime(liab["end_date"], errors="coerce")
            liab = liab.drop_duplicates(subset="end_date", keep="last")

            merged = df.merge(liab[["end_date", "total_liab"]], on="end_date", how="left")
            merged.to_parquet(path, index=False)
            ok += 1
        except Exception as e:
            print(f"  {ts_code} 失败: {e}")
            fail += 1

        time.sleep(DELAY)
        if i % 100 == 0:
            print(f"  进度：{i}/{len(files)}（成功 {ok}，跳过 {skip}，失败 {fail}）")

    print(f"完成：成功 {ok}，跳过（已有字段） {skip}，失败 {fail}")


if __name__ == "__main__":
    main()
