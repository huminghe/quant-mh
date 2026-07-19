"""
拉取ETF季度持仓明细（fund_portfolio），用于构建"ETF→申万一级行业"系统化映射。

背景：申万行业分层候选池方案需要知道每只ETF的成分股行业分布，
tushare的 fund_portfolio 接口返回季度持仓明细（symbol/stk_mkv_ratio等），
可据此把ETF持仓加权 stock_sw_industry.parquet 算出行业暴露。

已实测确认：
- 单次全历史请求（2016至今）不会像 fund_adj 那样静默截断，可直接一次性拉取
- 中国基金季报（Q1/Q3）只披露前十大持仓（stk_mkv_ratio 加总约50-58%），
  半年报/年报（Q2/Q4）才披露完整持仓（加总100%），下游做集中度计算时需注意
- QDII/跨境ETF（如159941纳指100、513500标普500）通常返回空，属正常情况

候选范围：复用 etf_all_candidates.parquet（431只机械化候选池，见
etf_rotation_v23_universe_bias_test.py 导出）。
"""

import pathlib
import time
import pandas as pd

from fetch_data import init_pro

DATA_DIR = pathlib.Path(__file__).parent
CANDIDATES_PATH = DATA_DIR / "etf_all_candidates.parquet"
CACHE_DIR = DATA_DIR / "fund_portfolio_cache"
START_DATE = "20160101"


def fetch_holdings_for_candidates(codes: list) -> None:
    pro = init_pro()
    today = pd.Timestamp.today().strftime("%Y%m%d")
    CACHE_DIR.mkdir(exist_ok=True)
    total = len(codes)
    n_empty = 0
    for i, code in enumerate(codes, 1):
        path = CACHE_DIR / f"{code}.parquet"
        if path.exists():
            continue
        for attempt in range(4):
            try:
                df = pro.fund_portfolio(ts_code=code, start_date=START_DATE, end_date=today)
                if df is not None and not df.empty:
                    df.to_parquet(path, index=False)
                    print(f"[{i:03d}/{total}] {code} 完成 {len(df)} 条，{df['end_date'].nunique()} 期")
                else:
                    # 空结果也要落一个空标记文件，避免下次重跑重复请求
                    pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "symbol",
                                          "mkv", "amount", "stk_mkv_ratio", "stk_float_ratio"]).to_parquet(path, index=False)
                    n_empty += 1
                    print(f"[{i:03d}/{total}] {code} 无持仓数据（QDII/跨境/新发行等）")
                break
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"[{i:03d}/{total}] {code} 第{attempt+1}次失败: {e}，{wait}s后重试")
                time.sleep(wait)
        time.sleep(0.3)
    print(f"\n完成，空持仓（含QDII）标的数：{n_empty}")


def main():
    candidates = pd.read_parquet(CANDIDATES_PATH)["ts_code"].tolist()
    print(f"候选池共 {len(candidates)} 只，开始拉取持仓（已缓存的跳过）...")
    fetch_holdings_for_candidates(candidates)


if __name__ == "__main__":
    main()
