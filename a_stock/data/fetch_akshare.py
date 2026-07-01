"""
akshare ETF 冷备下载
用途：tushare 不可用时，从 akshare 下载 ETF 历史日线并存为相同格式的 Parquet。
注意：akshare 来源是爬虫，不稳定，只用于应急恢复。正常情况用 fetch_data.py。
BaoStock 不支持 ETF 历史日线，故冷备改用 akshare。
"""

import time
import pathlib
import akshare as ak
import pandas as pd

from etf_universe import ETF_UNIVERSE, ETF_CODES

DATA_DIR = pathlib.Path(__file__).parent / "daily"
START_DATE = "20150101"

# ── 工具函数 ──────────────────────────────────────────────

def ts_to_ak_code(ts_code: str) -> str:
    """510300.SH → 510300，159915.SZ → 159915（akshare 只要纯数字代码）"""
    return ts_code.split(".")[0]


# ── 单只 ETF 下载 ─────────────────────────────────────────

def fetch_single_ak(ts_code: str, start_date: str = START_DATE, end_date: str = None) -> pd.DataFrame:
    """
    从 akshare 下载单只 ETF 后复权日线数据。
    返回列与 fetch_data.py 保持一致：trade_date, open, high, low, close, vol, amount
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    code = ts_to_ak_code(ts_code)
    # fund_etf_hist_em 是东方财富数据源
    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="hfq",   # 后复权
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # akshare 返回列：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    col_map = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "vol",
        "成交额": "amount",
    }
    df = df.rename(columns=col_map)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    keep = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["close"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


# ── 批量下载 ──────────────────────────────────────────────

def run_download(codes: list[str] = ETF_CODES, delay: float = 1.0) -> None:
    """
    从 akshare 下载所有 ETF 历史数据（覆盖写入，不做增量）。
    适合：tushare 不可用时的完整冷备恢复。
    delay: akshare 是爬虫，间隔 1s 防封。
    """
    today = pd.Timestamp.today().strftime("%Y%m%d")
    total = len(codes)

    print("akshare ETF 冷备下载（来源：东方财富，爬虫，不稳定）")
    print(f"共 {total} 只 ETF，数据目录：{DATA_DIR.resolve()}")
    print("-" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0

    for i, ts_code in enumerate(codes, 1):
        name = ETF_UNIVERSE.get(ts_code, ts_code)
        try:
            df = fetch_single_ak(ts_code, end_date=today)
            if df.empty:
                print(f"[{i:02d}/{total}] {ts_code} {name:<18} 无数据")
                fail += 1
            else:
                path = DATA_DIR / f"{ts_code}.parquet"
                df.to_parquet(path, index=False)
                ok += 1
                print(f"[{i:02d}/{total}] {ts_code} {name:<18} {len(df)} 条")
        except Exception as e:
            fail += 1
            print(f"[{i:02d}/{total}] {ts_code} {name:<18} 失败: {e}")

        time.sleep(delay)

    print("-" * 60)
    print(f"完成：成功 {ok} 只，失败/无数据 {fail} 只")
    if fail > 0:
        print("失败标的建议用 tushare 手动补全（fetch_data.py）")


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    run_download()
