"""
商品期货主力连续合约行情数据获取（fut_daily）

用于指数增强候选因子"商品期货价格动量/持仓变化→产业链行业传导"IC验证
（详见 a_stock/backtest/factor_ic_futures_commodity.py）。信号思路：原材料
期货（钢铁/有色/煤炭/石油石化/基础化工/建筑材料上游品种）价格动量和
持仓量变化领先对应申万一级行业股票收益（原材料成本/景气度传导逻辑）。

数据特性（已实测确认）：
- 主力连续合约用不带月份后缀的代码（如 RB.SHF、I.DCE），tushare 自动
  跟踪当前持仓量最大的具体合约，值与最大持仓合约完全一致（已核对
  RB.SHF vs RB2701.SHF 2026-09-08收盘价/持仓量一致）。次主力连续合约
  用 XXL 后缀（如 RBL.SHF），跟踪持仓量次大的合约（已核对随时间切换
  RB2606→RB2607→RB2609，验证为动态跟踪非固定次月）。
- fut_basic 接口在本账号权限下对多个交易所（至少SHF）返回空DataFrame，
  原因未查明；但不需要它——主力连续代码已经是可直接使用的连续序列，
  不需要 fut_basic 提供的合约元数据（乘数/上市日期等）。
- 单次调用硬上限2000条，2015年至今需要 offset 分页（2次调用覆盖，
  与 fund_adj/fut_settle 同类截断问题）。
- 没有独立的现货基差接口；本脚本只拉取主力连续合约收盘价+持仓量(oi)，
  信号计算（价格动量、持仓量变化率）在 factor_ic_futures_commodity.py
  里完成，不在本脚本内加工。

品种→申万一级行业映射（仅覆盖中证500，沪深300商品行业成分股仅42只
低于50只截面门槛，本次IC检验范围限定中证500，详见用户2026-09-09决策）：
  钢铁：RB.SHF（螺纹钢）、HC.SHF（热卷）、I.DCE（铁矿石，原料）
  有色金属：CU.SHF（铜）、AL.SHF（铝）、ZN.SHF（锌）
  煤炭：J.DCE（焦炭）、JM.DCE（焦煤）、ZC.ZCE（动力煤）
  石油石化：SC.INE（原油）、FU.SHF（燃料油）、BU.SHF（沥青）
  基础化工：MA.ZCE（甲醇）、TA.ZCE（PTA）、V.DCE（PVC）
  建筑材料：FG.ZCE（玻璃）、SA.ZCE（纯碱）

用法：
  cd a_stock/data
  python fetch_futures_commodity.py              # 全量下载主力连续合约
"""

import os
import time
import pathlib

import pandas as pd
import tushare as ts

DATA_DIR   = pathlib.Path(__file__).parent
OUT_FILE   = DATA_DIR / "futures_commodity_daily.parquet"
TOKEN_FILE = pathlib.Path.home() / ".tushare_token"

START_DATE = "20150101"
DELAY      = 0.35
BATCH_SIZE = 2000  # fut_daily 单次调用硬上限

# 品种 → 申万一级行业映射
COMMODITY_INDUSTRY = {
    "RB.SHF": "钢铁", "HC.SHF": "钢铁", "I.DCE": "钢铁",
    "CU.SHF": "有色金属", "AL.SHF": "有色金属", "ZN.SHF": "有色金属",
    "J.DCE": "煤炭", "JM.DCE": "煤炭", "ZC.ZCE": "煤炭",
    "SC.INE": "石油石化", "FU.SHF": "石油石化", "BU.SHF": "石油石化",
    "MA.ZCE": "基础化工", "TA.ZCE": "基础化工", "V.DCE": "基础化工",
    "FG.ZCE": "建筑材料", "SA.ZCE": "建筑材料",
}


def init_pro() -> ts.pro_api:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise ValueError("未找到 tushare token（环境变量 TUSHARE_TOKEN 或 ~/.tushare_token）")
    ts.set_token(token)
    return ts.pro_api()


def fetch_main_continuous(pro, ts_code: str) -> pd.DataFrame:
    """拉取主力连续合约全历史（offset分页处理2000条硬上限）"""
    frames = []
    offset = 0
    while True:
        df = pro.fut_daily(
            ts_code=ts_code, start_date=START_DATE, end_date="20261231",
            offset=offset,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount,oi",
        )
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < BATCH_SIZE:
            break
        offset += BATCH_SIZE

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    return result


def run_batch() -> None:
    pro = init_pro()
    codes = sorted(COMMODITY_INDUSTRY.keys())
    print(f"开始下载商品期货主力连续合约，共 {len(codes)} 个品种")

    frames = []
    for i, code in enumerate(codes, 1):
        df = fetch_main_continuous(pro, code)
        if df.empty:
            print(f"[{i:02d}/{len(codes)}] {code} 无数据")
            continue
        df["industry"] = COMMODITY_INDUSTRY[code]
        frames.append(df)
        print(f"[{i:02d}/{len(codes)}] {code}（{COMMODITY_INDUSTRY[code]}）"
              f"{len(df)}条  {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
        time.sleep(DELAY)

    if not frames:
        print("无任何数据，未保存")
        return

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(OUT_FILE, index=False)
    print(f"\n完成，共 {len(result)} 条，已保存至 {OUT_FILE}")


# ── 读取工具（供回测脚本调用）────────────────────────────

_futures_cache: pd.DataFrame | None = None


def load_futures_commodity() -> pd.DataFrame:
    """读取商品期货主力连续合约全量数据（含 industry 列）"""
    global _futures_cache
    if _futures_cache is not None:
        return _futures_cache
    if not OUT_FILE.exists():
        raise FileNotFoundError(f"未找到 {OUT_FILE}，请先运行 fetch_futures_commodity.py")
    df = pd.read_parquet(OUT_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    _futures_cache = df
    return df


if __name__ == "__main__":
    run_batch()
