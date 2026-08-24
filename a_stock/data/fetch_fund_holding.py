"""
公募主动管理基金持仓数据获取（fund_portfolio）

用于指数增强候选因子"基金持仓增减仓强度"（第十六轮候选①）的可行性调研
结论：候选池覆盖率不是瓶颈，但数据频率有限制——中国基金季报（Q1/Q3）
只披露前十大重仓股（stk_mkv_ratio加总约35-42%），半年报/年报（Q2/Q4）
才披露接近完整持仓（加总约60-65%，非100%，见fetch_etf_portfolio_holdings.py
同接口注释）。本脚本只拉取Q2/Q4记录，Q1/Q3记录不落盘，避免下游因子计算
把"仅前十大"和"接近完整"两种口径混用。

候选池：主动管理基金（fund_type∈{股票型,混合型} 且 invest_type不含"指数"
关键字，即剔除被动指数型/增强指数型），发行规模issue_amount>=2亿元，
覆盖沪深300+中证500全历史成分股所需的持仓视角（不是反过来按个股查，是
按基金查持仓明细再倒算个股层面的持仓强度，因为fund_portfolio只支持按
ts_code=基金代码查询，不支持按stk_code=个股反查）。

数据特性（已实测确认）：
- fund_portfolio 单次全历史请求（2016至今）不会像fund_adj那样静默截断
- Q1/Q3（一季报/三季报）只返回固定10条（前十大重仓），Q2/Q4（半年报/
  年报）返回40-90条不等（接近完整持仓）——本脚本按ann_date对应的
  end_date月份判断季度，只保留Q2(6月)/Q4(12月)记录
- 部分基金无持仓返回（新发行/清盘/QDII等），空结果也落一个空schema
  parquet占位，避免重跑重复请求

用法：
  cd a_stock/data
  python fetch_fund_holding.py               # 全量下载
  python fetch_fund_holding.py --min-issue 2  # 自定义发行规模门槛（亿元）
"""

import pathlib
import time
import argparse

import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
CACHE_DIR = DATA_DIR / "fund_holding_cache"
START_DATE = "20160101"
DEFAULT_MIN_ISSUE_AMOUNT = 2.0  # 亿元

EMPTY_SCHEMA = ["ts_code", "ann_date", "end_date", "symbol",
                "mkv", "amount", "stk_mkv_ratio", "stk_float_ratio"]


# ── 候选基金池 ────────────────────────────────────────────

def get_active_fund_pool(pro, min_issue_amount: float = DEFAULT_MIN_ISSUE_AMOUNT) -> list[str]:
    """
    主动管理基金池：fund_type∈{股票型,混合型}，剔除被动/增强指数型，
    发行规模>=min_issue_amount亿元，覆盖存续(L)+已清盘(D)（清盘基金在
    存续期内的历史持仓仍是有效point-in-time数据）。
    """
    frames = []
    for status in ["L", "D"]:
        df = pro.fund_basic(
            market="O", status=status,
            fields="ts_code,fund_type,invest_type,issue_amount",
        )
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    active = all_df[
        all_df["fund_type"].isin(["股票型", "混合型"])
        & ~all_df["invest_type"].isin(["被动指数型", "增强指数型"])
        & (all_df["issue_amount"] >= min_issue_amount)
    ]
    return sorted(active["ts_code"].unique())


# ── 数据获取（只保留Q2/Q4半年报/年报记录）───────────────────

def fetch_holding_for_fund(pro, ts_code: str) -> pd.DataFrame:
    today = pd.Timestamp.today().strftime("%Y%m%d")
    for attempt in range(4):
        try:
            df = pro.fund_portfolio(ts_code=ts_code, start_date=START_DATE, end_date=today)
            break
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"    {ts_code} 第{attempt+1}次失败: {e}，{wait}s后重试")
            time.sleep(wait)
    else:
        return pd.DataFrame(columns=EMPTY_SCHEMA)

    if df is None or df.empty:
        return pd.DataFrame(columns=EMPTY_SCHEMA)

    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df = df.dropna(subset=["end_date"])
    # 只保留半年报(6月)/年报(12月)记录，一季报/三季报只有前十大重仓，
    # 与半年报/年报的"接近完整持仓"口径不一致，不落盘避免下游混用
    df = df[df["end_date"].dt.month.isin([6, 12])]
    if df.empty:
        return pd.DataFrame(columns=EMPTY_SCHEMA)

    df["end_date"] = df["end_date"].dt.strftime("%Y%m%d")
    return df[EMPTY_SCHEMA]


def run_batch(codes: list[str]) -> None:
    pro = init_pro()
    CACHE_DIR.mkdir(exist_ok=True)
    total = len(codes)
    n_empty = 0
    for i, code in enumerate(codes, 1):
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            continue
        df = fetch_holding_for_fund(pro, code)
        df.to_parquet(path, index=False)
        if df.empty:
            n_empty += 1
        else:
            if i % 100 == 0 or i <= 3:
                print(f"[{i:04d}/{total}] {code} 完成 {len(df)} 条，{df['end_date'].nunique()} 期")
        time.sleep(0.3)
    print(f"\n完成，空持仓（含Q1/Q3-only/新发行/清盘等）标的数：{n_empty}/{total}")


# ── 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="公募主动管理基金持仓数据下载（fund_portfolio，仅Q2/Q4）")
    parser.add_argument("--min-issue", type=float, default=DEFAULT_MIN_ISSUE_AMOUNT,
                        help="最小发行规模（亿元），默认2亿")
    args = parser.parse_args()

    pro = init_pro()
    codes = get_active_fund_pool(pro, args.min_issue)
    print(f"主动管理基金候选池共 {len(codes)} 只（发行规模>={args.min_issue}亿），开始拉取持仓...")
    run_batch(codes)


if __name__ == "__main__":
    main()
