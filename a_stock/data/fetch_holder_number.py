"""
股东户数数据获取（stk_holdernumber，point-in-time）

用于指数增强候选因子"股东户数变化率"组合回测（详见
a_stock/backtest/factor_holder_number_backtest.py）。该因子已通过IC初筛
（ret20 IC均值0.0518, ret60 IC均值0.0439）和稳健性补测，本脚本负责拉取
组合回测所需的全历史数据。

数据特性（已实测确认）：
- 覆盖：中证500历史成分股100%可查，披露频率不定期——季度末为主，
  部分股票在年份/季度间额外披露过临时截面（如股权变动后），单只股票
  全历史约50-150条记录，单次调用不会遇到分页截断（已实测000001.SZ/
  600519.SH等全历史查询均一次性返回完整数据，不同于stk_holdertrade
  需要offset分页）。
- 同一end_date可能有多条公告（含更正公告），按ann_date取最后一条。
- holder_num字段偶发为NaN（尤其临时截面记录），下游计算需要dropna。
- 存储方式沿用 fetch_financials.py 的按股票分文件模式（point-in-time
  因子计算需要频繁按ann_date过滤单只股票的历史序列，与财务指标场景相同）。

用法：
  cd a_stock/data
  python fetch_holder_number.py               # 全量下载（沪深300+中证500历史成分股）
  python fetch_holder_number.py --update      # 增量更新（全量重拉单只股票，接口无法增量查询更正记录）
  python fetch_holder_number.py --index hs500 # 只下载中证500历史成分股（本因子只在中证500通过验证）
"""

import os
import sys
import time
import argparse
import pathlib

import pandas as pd
import tushare as ts

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR      = pathlib.Path(__file__).parent
HOLDER_NUM_DIR = DATA_DIR / "holder_number"
TOKEN_FILE    = pathlib.Path.home() / ".tushare_token"

START_DATE = "20140101"   # 与 fetch_financials.py 保持一致的历史起点
DELAY      = 0.35


# ── Token 初始化 ──────────────────────────────────────────

def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise ValueError("未找到 tushare token，请设置环境变量 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


# ── 数据获取 ──────────────────────────────────────────────

def fetch_holder_number(pro, ts_code: str) -> pd.DataFrame:
    """
    拉取单只股票的股东户数全历史。
    返回列：ts_code, ann_date, end_date, holder_num（已按end_date去重取最新公告）。
    """
    try:
        df = pro.stk_holdernumber(
            ts_code=ts_code,
            start_date=START_DATE,
            fields="ts_code,ann_date,end_date,holder_num",
        )
    except Exception as e:
        print(f"    {ts_code} API 错误: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["ann_date", "end_date", "holder_num"])

    # 同一end_date可能有多条公告（更正记录），取最新ann_date的一条
    df = (df.sort_values("ann_date")
            .drop_duplicates(subset=["ts_code", "end_date"], keep="last")
            .sort_values(["end_date", "ann_date"])
            .reset_index(drop=True))

    return df


# ── 批量下载 ──────────────────────────────────────────────

def get_hs500_codes() -> list[str]:
    """中证500历史成分股（本因子只在该指数通过IC验证）"""
    fpath = DATA_DIR / "hs500_members.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"未找到 {fpath}，请先运行 fetch_index_members.py --index hs500")
    df = pd.read_parquet(fpath)
    return sorted(df["con_code"].unique())


def get_all_codes(index: str) -> list[str]:
    if index == "hs500":
        return get_hs500_codes()
    if index == "all":
        codes = set()
        for fname in ["hs300_members.parquet", "hs500_members.parquet"]:
            fpath = DATA_DIR / fname
            if fpath.exists():
                codes.update(pd.read_parquet(fpath)["con_code"].unique())
        return sorted(codes)
    raise ValueError(f"未知 index={index}")


def update_stock_holder_number(pro, ts_code: str) -> str:
    """
    更新单只股票股东户数数据。
    策略：全量重拉（接口会返回历史更正记录，增量不可靠，同 fetch_financials.py）。
    """
    path = HOLDER_NUM_DIR / f"{ts_code}.parquet"

    if path.exists():
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s")
        if mtime.date() >= pd.Timestamp.today().date():
            return "已最新"

    df = fetch_holder_number(pro, ts_code)
    if df.empty:
        return "无数据"

    HOLDER_NUM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"保存 {len(df)} 条（{df['end_date'].min().strftime('%Y-%m')} ~ {df['end_date'].max().strftime('%Y-%m')}）"


def run_batch(codes: list[str], delay: float = DELAY) -> None:
    pro = init_pro()
    total = len(codes)
    print(f"开始下载股东户数数据，共 {total} 只股票")
    print(f"数据目录：{HOLDER_NUM_DIR.resolve()}")
    print("-" * 60)

    ok, skip, fail = 0, 0, 0
    for i, ts_code in enumerate(codes, 1):
        result = update_stock_holder_number(pro, ts_code)
        if result == "已最新":
            skip += 1
        elif result == "无数据":
            fail += 1
            if fail <= 10 or fail % 50 == 0:
                print(f"[{i:04d}/{total}] {ts_code} 无数据")
        else:
            ok += 1
            if ok % 100 == 0 or i <= 3:
                print(f"[{i:04d}/{total}] {ts_code} {result}")
        time.sleep(delay)

    print("-" * 60)
    print(f"完成：下载 {ok} 只，跳过 {skip} 只，失败/无数据 {fail} 只")


# ── 读取工具（供回测脚本调用）────────────────────────────

_holder_num_cache: dict[str, pd.DataFrame] = {}


def load_holder_number(ts_code: str) -> pd.DataFrame:
    """读取单只股票股东户数数据，按 ann_date 排序（point-in-time 时间轴）"""
    if ts_code in _holder_num_cache:
        return _holder_num_cache[ts_code]
    path = HOLDER_NUM_DIR / f"{ts_code}.parquet"
    if not path.exists():
        _holder_num_cache[ts_code] = pd.DataFrame()
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values("end_date").reset_index(drop=True)
    _holder_num_cache[ts_code] = df
    return df


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="股东户数数据下载（stk_holdernumber）")
    parser.add_argument("--update", action="store_true",
                        help="增量更新（跳过今天已下载的，接口无法真正增量查询）")
    parser.add_argument("--index", choices=["hs500", "all"], default="hs500",
                        help="下载哪个指数的历史成分股（默认中证500，本因子只在该指数通过IC验证）")
    parser.add_argument("--codes", nargs="+", metavar="CODE",
                        help="指定股票代码，如 600519.SH 000858.SZ")
    args = parser.parse_args()

    if args.codes:
        codes = args.codes
    else:
        codes = get_all_codes(args.index)
        print(f"{args.index} 历史成分股共 {len(codes)} 只")

    run_batch(codes)


if __name__ == "__main__":
    main()
