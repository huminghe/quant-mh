"""
财务指标数据获取（用于质量/估值因子）
- 拉取 tushare fina_indicator：roe_dt（ROE_TTM）、eps（EPS）、ann_date（公告日）
- point-in-time：用 ann_date 而非 end_date，避免前视偏差
- 存储：a_stock/data/financials/{ts_code}.parquet

用法：
  cd a_stock/data
  python fetch_financials.py             # 全量下载（沪深300+中证500历史成分股）
  python fetch_financials.py --update    # 增量更新
  python fetch_financials.py --codes 600519.SH 000858.SZ  # 指定股票
"""

import os
import sys
import time
import argparse
import pathlib

import pandas as pd
import tushare as ts

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR       = pathlib.Path(__file__).parent
FINANCIALS_DIR = DATA_DIR / "financials"
TOKEN_FILE     = pathlib.Path.home() / ".tushare_token"

START_DATE = "20140101"   # 财务数据从2014年开始，覆盖2015+的因子计算
DELAY      = 0.4          # API 请求间隔（秒）

# 需要的财务字段
FIELDS = ("ts_code,ann_date,end_date,roe_dt,eps,netprofit_margin,debt_to_assets,"
          "current_ratio,ocf_to_profit,netprofit_yoy")


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

def fetch_fina_indicator(pro, ts_code: str) -> pd.DataFrame:
    """
    拉取单只股票的财务指标（所有报告期）。
    返回列：ts_code, ann_date, end_date, roe_dt, eps, netprofit_margin,
            debt_to_assets, current_ratio
    """
    try:
        df = pro.fina_indicator(
            ts_code=ts_code,
            start_date=START_DATE,
            fields=FIELDS,
        )
    except Exception as e:
        print(f"    {ts_code} API 错误: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 日期列统一转 datetime
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # 去重：同一 end_date 可能有多次公告（修正公告），取最新公告日的
    df = (df.sort_values("ann_date")
            .drop_duplicates(subset=["ts_code", "end_date"], keep="last")
            .sort_values(["end_date", "ann_date"])
            .reset_index(drop=True))

    return df


def get_last_end_date(ts_code: str) -> str | None:
    """读取已有数据的最后报告期，用于增量更新"""
    path = FINANCIALS_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["end_date"])
    if df.empty:
        return None
    return pd.Timestamp(df["end_date"].max()).strftime("%Y%m%d")


def update_stock_financials(pro, ts_code: str) -> str:
    """
    增量更新单只股票财务数据。
    策略：全量重拉（财务数据有追溯修正，增量不可靠），但限制频率。
    """
    path = FINANCIALS_DIR / f"{ts_code}.parquet"

    # 若文件已存在且是今天以内，跳过
    if path.exists():
        # 若文件已存在但缺少新字段（ocf_to_profit/netprofit_yoy），强制重拉
        try:
            existing = pd.read_parquet(path, columns=["ann_date"])
            need_refresh = False
            # 检查是否缺少新字段
            all_cols = pd.read_parquet(path).columns.tolist()
            if "ocf_to_profit" not in all_cols or "netprofit_yoy" not in all_cols:
                need_refresh = True
        except Exception:
            need_refresh = True

        if not need_refresh:
            mtime = pd.Timestamp(path.stat().st_mtime, unit="s")
            if mtime.date() >= pd.Timestamp.today().date():
                return "已最新"

    df = fetch_fina_indicator(pro, ts_code)
    if df.empty:
        return "无数据"

    FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"保存 {len(df)} 条（{df['end_date'].min().strftime('%Y-%m')} ~ {df['end_date'].max().strftime('%Y-%m')}）"


# ── 批量下载 ──────────────────────────────────────────────

def get_all_codes() -> list[str]:
    """合并沪深300 + 中证500历史成分股"""
    codes = set()
    for fname in ["hs300_members.parquet", "hs500_members.parquet"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            codes.update(df["con_code"].unique())
    return sorted(codes)


def run_batch(codes: list[str], delay: float = DELAY) -> None:
    pro = init_pro()
    total = len(codes)
    print(f"开始下载财务指标，共 {total} 只股票")
    print(f"数据目录：{FINANCIALS_DIR.resolve()}")
    print("-" * 60)

    ok, skip, fail = 0, 0, 0
    for i, ts_code in enumerate(codes, 1):
        result = update_stock_financials(pro, ts_code)
        if result == "已最新":
            skip += 1
        elif result == "无数据":
            fail += 1
            if fail <= 10 or fail % 50 == 0:
                print(f"[{i:04d}/{total}] {ts_code} 无数据")
        else:
            ok += 1
            if ok % 50 == 0 or i <= 3:
                print(f"[{i:04d}/{total}] {ts_code} {result}")
        time.sleep(delay)

    print("-" * 60)
    print(f"完成：下载 {ok} 只，跳过 {skip} 只，失败 {fail} 只")


# ── 读取工具（供因子脚本调用）────────────────────────────

def load_financials(ts_code: str) -> pd.DataFrame:
    """
    读取单只股票财务数据，按 ann_date 排序。
    用 ann_date 作为 point-in-time 时间轴（公告日后才可知）。
    """
    path = FINANCIALS_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df.sort_values("ann_date").reset_index(drop=True)


def get_factor_pit(ts_code: str, as_of_date: pd.Timestamp,
                   field: str) -> float | None:
    """
    获取截至 as_of_date 可知的最新财务指标值（point-in-time）。
    只取 ann_date <= as_of_date 的最新记录。
    """
    df = load_financials(ts_code)
    if df.empty:
        return None
    valid = df[df["ann_date"] <= as_of_date]
    if valid.empty:
        return None
    return valid.iloc[-1][field]


def build_factor_panel_pit(
    codes: list[str],
    dates: list[pd.Timestamp],
    field: str,
    min_coverage: float = 0.3,
) -> pd.DataFrame:
    """
    构建因子面板（宽格式）：index=date，columns=ts_code。
    每个日期取各股 point-in-time 的最新财务值。

    min_coverage: 截面有效股票比例阈值（低于则该日期全填 NaN）
    """
    records = {}
    for code in codes:
        df = load_financials(code)
        if df.empty:
            continue
        df = df[["ann_date", field]].dropna()
        # 对每个 date，取 ann_date <= date 的最新值
        values = {}
        for d in dates:
            valid = df[df["ann_date"] <= d]
            if not valid.empty:
                values[d] = valid.iloc[-1][field]
        records[code] = values

    if not records:
        return pd.DataFrame(index=dates)

    panel = pd.DataFrame(records, index=dates)

    # 过滤有效截面比例不足的日期
    coverage = panel.notna().sum(axis=1) / len(panel.columns)
    panel.loc[coverage < min_coverage] = float("nan")

    return panel


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="财务指标数据下载（ROE_TTM/EPS等）")
    parser.add_argument("--update", action="store_true",
                        help="增量更新（跳过今天已下载的）")
    parser.add_argument("--codes", nargs="+", metavar="CODE",
                        help="指定股票代码，如 600519.SH 000858.SZ")
    args = parser.parse_args()

    if args.codes:
        codes = args.codes
    else:
        codes = get_all_codes()
        if not codes:
            print("未找到成分股快照，请先运行 fetch_index_members.py")
            sys.exit(1)
        print(f"沪深300 + 中证500 历史成分股共 {len(codes)} 只")

    run_batch(codes)


if __name__ == "__main__":
    main()
