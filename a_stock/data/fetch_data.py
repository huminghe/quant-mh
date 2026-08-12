"""
tushare 数据获取主入口
- 每次运行：全量重新拉取 2015-01-01 至今的完整历史数据并覆盖本地文件
  （不用增量追加，因为后复权基准会随分红/拆分变化，增量拼接会产生价格断层）
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

def fetch_fund_adj_full(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    分段拉取 fund_adj 复权因子，覆盖完整日期范围。
    fund_adj 单次调用有约2000条上限，长区间（如2015年至今）会静默截断到最近的2000条，
    早期日期拿不到 adj_factor，如果不分段会导致早期价格漏复权、在截断点产生价格断层。
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    chunks = []
    cur = start_ts
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=2) - pd.Timedelta(days=1), end_ts)
        part = pro.fund_adj(
            ts_code=ts_code,
            start_date=cur.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
        )
        if part is not None and not part.empty:
            chunks.append(part)
        cur = chunk_end + pd.Timedelta(days=1)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates("trade_date")


def fetch_single(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 tushare 获取单只 ETF 复权日线数据。
    返回列：trade_date, open, high, low, close, vol, amount
    trade_date 为 datetime 类型，按升序排列。

    注意：不用 pro_bar(adj="hfq")！实测发现该接口在 ETF 份额折算（拆分/合并）
    事件上不做复权修正，会在事件发生日产生价格断层（实测最大达-90%）。
    改为拉取不复权价 + fund_adj 复权因子，手工相乘还原正确的后复权序列。
    """
    df = ts.pro_bar(
        ts_code=ts_code,
        api=pro,
        asset="FD",          # FD = 基金/ETF
        adj=None,            # 不复权，自己用 adj_factor 手工复权
        start_date=start_date,
        end_date=end_date,
        factors=["tor"],     # 换手率，轮动策略用得上
    )
    if df is None or df.empty:
        return pd.DataFrame()

    adj = fetch_fund_adj_full(pro, ts_code, start_date, end_date)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    if adj is not None and not adj.empty:
        adj = adj[["trade_date", "adj_factor"]].copy()
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        df = df.merge(adj, on="trade_date", how="left")
        df["adj_factor"] = df["adj_factor"].fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * df["adj_factor"]
        df = df.drop(columns=["adj_factor"])

    # 只保留需要的列，统一字段名
    keep = ["trade_date", "open", "high", "low", "close", "vol", "amount", "tor"]
    df = df[[c for c in keep if c in df.columns]]
    return df


# ── 全量更新逻辑 ──────────────────────────────────────────
#
# 注意：不能用"增量追加"！后复权（adj="hfq"）的价格基准会在每次分红/拆分
# 发生后被 tushare 重新计算，增量拉取的新数据和本地缓存的旧数据基准不一致，
# 拼接后会在复权事件发生日产生价格断层（实测最大达-90%），严重扭曲回测。
# 因此每次更新都全量重新拉取完整历史，保证同一基准。

def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def update_single(pro, ts_code: str, today: str, stale_days: int = 3) -> str:
    """
    全量更新单只 ETF（重新拉取完整历史并覆盖本地文件）。
    若本地缓存已存在且最新数据距今不超过 stale_days 天，跳过不请求，
    避免候选池扩容到431只后每次全量刷新触发 tushare 频率限制（500次/分钟，
    fund_adj 分段拉取会让单只ETF产生多次调用）。
    返回操作描述：'更新 N 条'、'无数据'、'已是最新，跳过'
    """
    parquet_path = DATA_DIR / f"{ts_code}.parquet"
    if parquet_path.exists():
        try:
            cached = pd.read_parquet(parquet_path, columns=["trade_date"])
            last_date = pd.to_datetime(cached["trade_date"]).max()
            if last_date >= pd.Timestamp.today() - pd.Timedelta(days=stale_days):
                return "已是最新，跳过"
        except Exception:
            pass  # 读取失败视为需要重新拉取

    new_df = fetch_single(pro, ts_code, START_DATE, today)
    if new_df.empty:
        return "无数据"

    save_parquet(new_df, parquet_path)
    return f"更新 {len(new_df)} 条"


# ── 批量更新 ──────────────────────────────────────────────

def run_update(codes: list[str] = ETF_CODES, delay: float = 0.4) -> None:
    """
    批量增量更新所有 ETF（跳过近期已刷新过的，见 update_single 的 stale_days 检查）。
    delay：每次请求间隔（秒），tushare 5000积分 500次/分钟，0.4s 约150次/分钟，安全。
    """
    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")
    total = len(codes)

    print(f"开始更新，共 {total} 只 ETF，截止日期 {today}")
    print(f"数据目录：{DATA_DIR.resolve()}")
    print("-" * 60)

    ok, fail = 0, 0
    for i, ts_code in enumerate(codes, 1):
        name = ETF_UNIVERSE.get(ts_code, ts_code)
        try:
            result = update_single(pro, ts_code, today)
            ok += 1
            print(f"[{i:02d}/{total}] {ts_code} {name:<18} {result}")
        except Exception as e:
            fail += 1
            print(f"[{i:02d}/{total}] {ts_code} {name:<18} 失败: {e}")

        time.sleep(delay)

    print("-" * 60)
    print(f"完成：更新 {ok} 只，失败 {fail} 只")


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
