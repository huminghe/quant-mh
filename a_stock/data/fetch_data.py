"""
tushare 数据获取主入口
- 首次运行：下载 2015-01-01 至今的完整历史数据
- 增量更新：只补全缺失的最新交易日
- 存储格式：Parquet，每只 ETF 一个文件，按 trade_date 升序
"""

import os
import time
import getpass
import pathlib
import pandas as pd
import tushare as ts

from etf_universe import ETF_UNIVERSE, ETF_CODES

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "daily"
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"
START_DATE = "20150101"

# ── Token 管理 ────────────────────────────────────────────

def get_token() -> str:
    """按优先级读取 tushare token：环境变量 > 本地文件 > 交互输入"""
    # 1. 环境变量
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    # 2. 本地文件
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token

    # 3. 交互输入并保存
    print("未找到 tushare token，请输入（不会显示在终端）：")
    token = getpass.getpass("token > ").strip()
    if not token:
        raise ValueError("token 为空，无法继续")
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    print(f"token 已保存到 {TOKEN_FILE}")
    return token


def init_pro() -> ts.pro_api:
    token = get_token()
    ts.set_token(token)
    return ts.pro_api()


# ── 单只 ETF 数据获取 ─────────────────────────────────────

def fetch_single(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 tushare 获取单只 ETF 复权日线数据。
    返回列：trade_date, open, high, low, close, vol, amount
    trade_date 为 datetime 类型，按升序排列。
    """
    df = ts.pro_bar(
        ts_code=ts_code,
        api=pro,
        asset="FD",          # FD = 基金/ETF
        adj="hfq",           # 后复权
        start_date=start_date,
        end_date=end_date,
        factors=["tor"],     # 换手率，轮动策略用得上
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 只保留需要的列，统一字段名
    keep = ["trade_date", "open", "high", "low", "close", "vol", "amount", "tor"]
    df = df[[c for c in keep if c in df.columns]]
    return df


# ── 增量更新逻辑 ──────────────────────────────────────────

def get_last_date(parquet_path: pathlib.Path) -> str | None:
    """读取已有 Parquet 文件的最后一条 trade_date，返回 yyyymmdd 字符串"""
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path, columns=["trade_date"])
    if df.empty:
        return None
    last = df["trade_date"].max()
    return pd.Timestamp(last).strftime("%Y%m%d")


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def update_single(pro, ts_code: str, today: str) -> str:
    """
    增量更新单只 ETF。
    返回操作描述：'新建'、'更新 N 条'、'已最新'、'无数据'
    """
    parquet_path = DATA_DIR / f"{ts_code}.parquet"
    last_date = get_last_date(parquet_path)

    if last_date is None:
        start = START_DATE
        action = "新建"
    elif last_date >= today:
        return "已最新"
    else:
        # 从上次最后日期的下一天开始拉
        start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        action = "更新"

    new_df = fetch_single(pro, ts_code, start, today)
    if new_df.empty:
        return "无数据"

    if action == "新建":
        save_parquet(new_df, parquet_path)
        return f"新建 {len(new_df)} 条"
    else:
        # 追加到已有数据
        old_df = pd.read_parquet(parquet_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
        save_parquet(merged, parquet_path)
        return f"更新 {len(new_df)} 条"


# ── 批量更新 ──────────────────────────────────────────────

def run_update(codes: list[str] = ETF_CODES, delay: float = 0.4) -> None:
    """
    批量增量更新所有 ETF。
    delay：每次请求间隔（秒），tushare 5000积分 500次/分钟，0.4s 约150次/分钟，安全。
    """
    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")
    total = len(codes)

    print(f"开始更新，共 {total} 只 ETF，截止日期 {today}")
    print(f"数据目录：{DATA_DIR.resolve()}")
    print("-" * 60)

    ok, skip, fail = 0, 0, 0
    for i, ts_code in enumerate(codes, 1):
        name = ETF_UNIVERSE.get(ts_code, ts_code)
        try:
            result = update_single(pro, ts_code, today)
            if result == "已最新":
                skip += 1
            else:
                ok += 1
            print(f"[{i:02d}/{total}] {ts_code} {name:<18} {result}")
        except Exception as e:
            fail += 1
            print(f"[{i:02d}/{total}] {ts_code} {name:<18} 失败: {e}")

        time.sleep(delay)

    print("-" * 60)
    print(f"完成：更新 {ok} 只，已最新 {skip} 只，失败 {fail} 只")


# ── 读取工具函数（供回测脚本调用）────────────────────────

def load_etf(ts_code: str) -> pd.DataFrame:
    """读取单只 ETF 本地数据"""
    path = DATA_DIR / f"{ts_code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{ts_code} 数据文件不存在，请先运行 fetch_data.py")
    return pd.read_parquet(path)


def load_close_matrix(codes: list[str] = ETF_CODES) -> pd.DataFrame:
    """
    读取多只 ETF 的收盘价矩阵。
    返回：index=trade_date，columns=ts_code
    """
    frames = {}
    for code in codes:
        path = DATA_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "close"])
            frames[code] = df.set_index("trade_date")["close"]

    if not frames:
        raise FileNotFoundError("没有找到任何本地数据，请先运行 fetch_data.py")

    matrix = pd.DataFrame(frames)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index()
    # 截断末尾数据不完整的行（避免部分ETF数据滞后导致净值计算异常）
    # 保留超过半数标的有数据的最后一天
    last_complete = matrix.index[matrix.notna().sum(axis=1) >= len(matrix.columns) // 2][-1]
    return matrix.loc[:last_complete]


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    run_update()
