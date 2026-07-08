"""
沪深 300 成分股日线数据获取
- 拉取历史成分股变动（index_weight），按月快照构建 point-in-time 成分股列表
- 下载所有出现过的成分股后复权日线数据，存 Parquet
- 增量更新：只补全缺失的最新交易日

目录结构：
  a_stock/data/stock_daily/{ts_code}.parquet   -- 个股日线
  a_stock/data/hs300_members.parquet           -- 每月成分股快照（point-in-time）

用法：
  cd a_stock/data
  python fetch_index_members.py             # 首次全量下载
  python fetch_index_members.py --update    # 增量更新
"""

import os
import sys
import time
import argparse
import pathlib
import getpass

import numpy as np
import pandas as pd
import tushare as ts

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR    = pathlib.Path(__file__).parent
STOCK_DIR   = DATA_DIR / "stock_daily"
MEMBERS_FILE = DATA_DIR / "hs300_members.parquet"
TOKEN_FILE  = pathlib.Path.home() / ".tushare_token"

INDEX_CODE  = "000300.SH"   # 沪深 300
START_DATE  = "20150101"
DELAY       = 0.35           # 每次 API 请求间隔（秒）


# ── Token / API 初始化 ────────────────────────────────────

def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        print("未找到 tushare token，请输入（不会显示在终端）：")
        token = getpass.getpass("token > ").strip()
        if not token:
            raise ValueError("token 为空")
        TOKEN_FILE.write_text(token)
        TOKEN_FILE.chmod(0o600)
    ts.set_token(token)
    return ts.pro_api()


# ── Step 1：拉取历史成分股快照 ────────────────────────────

def fetch_index_members_history(pro) -> pd.DataFrame:
    """
    拉取沪深 300 历史成分股月度权重快照（index_weight）。
    返回：trade_date（月末日期）、con_code（成分股代码）两列。
    tushare index_weight 每月一条快照，需要循环按月拉取。
    """
    print("拉取沪深 300 历史成分股快照...")

    today = pd.Timestamp.today()
    # 生成从 START_DATE 到今天的每月最后一天列表
    months = pd.date_range(
        start=START_DATE, end=today, freq="ME"
    )

    all_records = []
    for i, month_end in enumerate(months):
        date_str = month_end.strftime("%Y%m%d")
        try:
            df = pro.index_weight(
                index_code=INDEX_CODE,
                trade_date=date_str,
                fields="trade_date,con_code"
            )
            if df is not None and not df.empty:
                all_records.append(df)
                print(f"  {date_str}: {len(df)} 只成分股")
            else:
                print(f"  {date_str}: 无数据（可能非交易日）")
        except Exception as e:
            print(f"  {date_str}: 失败 - {e}")
        time.sleep(DELAY)

    if not all_records:
        raise RuntimeError("未能拉取到任何成分股数据，请检查 token 积分是否充足")

    members = pd.concat(all_records, ignore_index=True)
    members["trade_date"] = pd.to_datetime(members["trade_date"])
    members = members.drop_duplicates().sort_values(["trade_date", "con_code"])
    return members


# ── Step 2：保存成分股快照 ────────────────────────────────

def save_members(members: pd.DataFrame) -> None:
    MEMBERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    members.to_parquet(MEMBERS_FILE, index=False)
    n_dates = members["trade_date"].nunique()
    n_stocks = members["con_code"].nunique()
    print(f"成分股快照已保存：{n_dates} 个月份，共 {n_stocks} 只历史成分股")


# ── Step 3：获取所有历史成分股列表 ───────────────────────

def get_all_member_codes() -> list[str]:
    """从已保存快照中提取所有出现过的成分股代码"""
    if not MEMBERS_FILE.exists():
        raise FileNotFoundError("成分股快照不存在，请先运行 fetch_index_members_history()")
    members = pd.read_parquet(MEMBERS_FILE)
    return sorted(members["con_code"].unique().tolist())


# ── Step 4：下载个股日线数据 ──────────────────────────────

def fetch_stock_daily(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取单只股票后复权日线数据。
    返回列：trade_date, open, high, low, close, vol, amount, adj_factor
    """
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            api=pro,
            asset="E",       # E = 股票
            adj="hfq",       # 后复权
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        print(f"    {ts_code} API 错误: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    keep = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def get_stock_last_date(ts_code: str) -> str | None:
    """读取已有 Parquet 文件的最后交易日"""
    path = STOCK_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["trade_date"])
    if df.empty:
        return None
    return pd.Timestamp(df["trade_date"].max()).strftime("%Y%m%d")


def update_stock(pro, ts_code: str, today: str) -> str:
    """增量更新单只股票日线数据"""
    path = STOCK_DIR / f"{ts_code}.parquet"
    last_date = get_stock_last_date(ts_code)

    if last_date is None:
        start = START_DATE
        action = "新建"
    elif last_date >= today:
        return "已最新"
    else:
        start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        action = "更新"

    new_df = fetch_stock_daily(pro, ts_code, start, today)
    if new_df.empty:
        return "无数据"

    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    if action == "新建":
        new_df.to_parquet(path, index=False)
        return f"新建 {len(new_df)} 条"
    else:
        old_df = pd.read_parquet(path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = (merged
                  .drop_duplicates("trade_date")
                  .sort_values("trade_date")
                  .reset_index(drop=True))
        merged.to_parquet(path, index=False)
        return f"更新 {len(new_df)} 条"


# ── Step 5：批量更新所有成分股 ────────────────────────────

def run_update_stocks(codes: list[str] = None, delay: float = DELAY) -> None:
    pro = init_pro()
    if codes is None:
        codes = get_all_member_codes()

    today = pd.Timestamp.today().strftime("%Y%m%d")
    total = len(codes)
    print(f"\n开始下载个股日线，共 {total} 只，截止日期 {today}")
    print(f"数据目录：{STOCK_DIR.resolve()}")
    print("-" * 60)

    ok, skip, fail = 0, 0, 0
    for i, ts_code in enumerate(codes, 1):
        try:
            result = update_stock(pro, ts_code, today)
            if result == "已最新":
                skip += 1
            elif result == "无数据":
                fail += 1
                print(f"[{i:04d}/{total}] {ts_code} 无数据")
            else:
                ok += 1
                if ok % 10 == 0 or i <= 5:
                    print(f"[{i:04d}/{total}] {ts_code} {result}")
        except Exception as e:
            fail += 1
            print(f"[{i:04d}/{total}] {ts_code} 失败: {e}")
        time.sleep(delay)

    print("-" * 60)
    print(f"完成：更新 {ok} 只，已最新 {skip} 只，失败/无数据 {fail} 只")


# ── 读取工具（供回测脚本调用）────────────────────────────

def load_stock(ts_code: str) -> pd.DataFrame:
    """读取单只股票本地数据"""
    path = STOCK_DIR / f"{ts_code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{ts_code} 数据不存在，请先运行 fetch_index_members.py")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date").sort_index()


def load_close_panel(codes: list[str] = None, min_coverage: float = 0.5) -> pd.DataFrame:
    """
    读取多只股票的收盘价面板（宽格式）。
    index=trade_date，columns=ts_code。
    截断末尾数据不完整的行（同 ETF 版本的 load_close_matrix）。
    """
    if codes is None:
        codes = get_all_member_codes()

    frames = {}
    for code in codes:
        path = STOCK_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames[code] = df.set_index("trade_date")["close"]

    if not frames:
        raise FileNotFoundError("没有找到任何股票数据，请先运行 fetch_index_members.py")

    panel = pd.DataFrame(frames).sort_index()

    # 截断末尾数据不完整的行
    threshold = int(len(panel.columns) * min_coverage)
    last_complete = panel.index[panel.notna().sum(axis=1) >= threshold][-1]
    return panel.loc[:last_complete]


def load_members_pit(date: pd.Timestamp, members_file: pathlib.Path = None) -> list[str]:
    """
    返回指定日期的 point-in-time 成分股列表。
    取 date 之前最近一个月末快照（避免前视偏差）。
    members_file: 成分股快照路径，默认为沪深300（hs300_members.parquet）
    """
    if members_file is None:
        members_file = MEMBERS_FILE
    members = pd.read_parquet(members_file)
    members["trade_date"] = pd.to_datetime(members["trade_date"])
    # 取 <= date 的最近快照
    valid = members[members["trade_date"] <= date]
    if valid.empty:
        return []
    latest = valid["trade_date"].max()
    return members[members["trade_date"] == latest]["con_code"].tolist()


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="沪深300成分股数据下载")
    parser.add_argument("--update", action="store_true",
                        help="只更新已有数据，不重新拉成分股快照")
    parser.add_argument("--members-only", action="store_true",
                        help="只更新成分股快照，不下载个股数据")
    args = parser.parse_args()

    pro = init_pro()

    if not args.update:
        # 全量：先拉成分股快照
        members = fetch_index_members_history(pro)
        save_members(members)

    if not args.members_only:
        # 下载/更新个股日线
        codes = get_all_member_codes()
        print(f"\n共 {len(codes)} 只历史成分股需要下载")
        run_update_stocks(codes)


if __name__ == "__main__":
    main()
