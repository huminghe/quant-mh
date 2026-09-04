"""
指数样本股调整效应实盘信号（中证500，仅调入组，T+1建仓-生效日前1交易日退出）

背景：第九轮已验证该效应通过完整组合回测（中证500，净超额+1.11%，p=0.003，
容量17.48亿元/次远超所需资金），详见research_index_enhancement.md。本脚本是实盘信号生成器，
读取fetch_rebalance_announcement.py抓到的调整事件（rebalance_pending.parquet），
按状态机推进"检测公告→提醒建仓→持仓等待→提醒退出→完成"四个阶段，本地log
记录状态避免重复提示或错过退出时点。

资金规划：本策略资金与ETF轮动10%仓位完全独立，单独切一块固定资金
（CAPITAL_YUAN，百万级资金的5%-10%区间取中值），常态空仓，每年6月/12月
各占用约9-10个交易日。

持仓精简：中证500每次调入组约50只，等权分散在几万元资金下单只买不到一手，
按公告日前20个交易日成交额精简为流动性最好的TOP_N只（复用
event_index_rebalance.py的load_daily_amount，DRY）。

运行方式：低频事件（2次/年），人工每天或每周手动跑一次查看状态即可，
不需要定时任务。跑之前需先用fetch_rebalance_announcement.py检查是否有
新公告，再用fetch_index_members.py --update更新个股行情到最新（本脚本
只读本地数据，不负责拉新数据，职责边界见.claude/rules/file-boundaries.md）。

用法：
  cd a_stock/backtest
  python index_rebalance_signal.py
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_index_members import init_pro, DATA_DIR

from event_index_rebalance import load_daily_amount, load_stock_close, shift_trading_day, get_effective_date

CAPITAL_YUAN = 80_000  # 本策略专用固定资金，百万级资金5%-10%区间取中值，可按实际调整
TOP_N = 6              # 精简持仓只数（按公告日前流动性排序取前N只）
LOT_SIZE = 100         # A股最小交易单位：1手=100股
LIQUIDITY_LOOKBACK = 20  # 选股用的成交额回看窗口（公告日前N个交易日）

PENDING_FILE = DATA_DIR / "rebalance_pending.parquet"
SIGNAL_LOG = pathlib.Path(__file__).parent / "results" / "index_rebalance_signal_log.csv"

STATUS_WAITING_ENTRY = "等待建仓"
STATUS_HOLDING = "持仓中"
STATUS_DONE = "已完成"


def load_trade_days(pro) -> pd.DatetimeIndex:
    cal = pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20271231", is_open="1")
    return pd.to_datetime(sorted(cal["cal_date"].tolist()))


def get_effective_date_after(trade_days: pd.DatetimeIndex, publish_date: pd.Timestamp) -> pd.Timestamp:
    """
    根据公告日期推算最近的生效日。中证指数公司规则是生效日前约2周发布
    公告，生效日通常落在公告日所在月或下一个月，遍历公告日之后0-2个月的
    候选月份，取第一个"当月第二个周五下一交易日"晚于publish_date的日期。
    """
    for offset_months in range(0, 3):
        base = publish_date + pd.DateOffset(months=offset_months)
        eff = get_effective_date(trade_days, base.year, base.month)
        if eff is not None and eff > publish_date:
            return eff
    return None


def pick_top_n_by_liquidity(codes: list[str], trade_days: pd.DatetimeIndex,
                             publish_date: pd.Timestamp, n: int) -> list[str]:
    """按公告日前LIQUIDITY_LOOKBACK个交易日的平均成交额排序，取流动性最好的前n只"""
    window_start = shift_trading_day(trade_days, publish_date, -LIQUIDITY_LOOKBACK)
    scores = {}
    for code in codes:
        s = load_daily_amount(code)
        s = s.loc[window_start:publish_date] if window_start is not None else s.loc[:publish_date]
        if not s.empty:
            scores[code] = s.mean()
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [code for code, _ in ranked[:n]]


def suggested_shares(code: str, capital_per_stock: float) -> int:
    """建议买入股数，按最新收盘价估算并向下取整到1手（100股）的整数倍"""
    s = load_stock_close(code)
    if s.empty:
        return 0
    latest_close = s.iloc[-1]
    shares = int(capital_per_stock // latest_close // LOT_SIZE) * LOT_SIZE
    return shares


# ── 本地状态log ──────────────────────────────────────────────

def load_log() -> pd.DataFrame:
    if not SIGNAL_LOG.exists():
        return pd.DataFrame(columns=["ann_id", "publish_date", "effective_date",
                                      "entry_date", "exit_date", "target_codes", "status"])
    return pd.read_csv(SIGNAL_LOG)


def save_log(log: pd.DataFrame) -> None:
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(SIGNAL_LOG, index=False)


def get_active_row(log: pd.DataFrame) -> pd.Series | None:
    """取log中尚未完成的事件（同一时刻只应有一条，2次/年事件不会重叠）"""
    active = log[log["status"] != STATUS_DONE]
    if active.empty:
        return None
    return active.iloc[-1]


# ── 主流程 ────────────────────────────────────────────────────

def register_new_event(pro, trade_days: pd.DatetimeIndex, log: pd.DataFrame) -> pd.DataFrame:
    """检查PENDING_FILE是否有log里还没记录过的新调整事件，有则登记"""
    if not PENDING_FILE.exists():
        print("空仓，无调整事件，下次公告发布后运行fetch_rebalance_announcement.py + 本脚本会自动检测")
        return log

    pending = pd.read_parquet(PENDING_FILE)
    if pending.empty:
        print("空仓，无调整事件")
        return log

    row = pending.iloc[-1]
    ann_id = str(row["ann_id"])
    if ann_id in log["ann_id"].astype(str).values:
        print("空仓，无新调整事件（该公告已处理过）")
        return log

    added = [c for c in str(row["added"]).split(",") if c]
    if not added:
        print("最新公告无调入个股（可能是纯调出或解析异常），跳过")
        return log

    publish_date = pd.Timestamp(row["publish_date"])
    eff_date = get_effective_date_after(trade_days, publish_date)
    if eff_date is None:
        print(f"无法推算公告[{row['title']}]对应的生效日，需人工核实")
        return log

    entry_date = shift_trading_day(trade_days, publish_date, 1)  # T+1建仓
    exit_date = shift_trading_day(trade_days, eff_date, -1)      # 生效日前1个交易日退出
    targets = pick_top_n_by_liquidity(added, trade_days, publish_date, TOP_N)

    new_row = {
        "ann_id": ann_id, "publish_date": publish_date.date(), "effective_date": eff_date.date(),
        "entry_date": entry_date.date() if entry_date is not None else None,
        "exit_date": exit_date.date() if exit_date is not None else None,
        "target_codes": ",".join(targets), "status": STATUS_WAITING_ENTRY,
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)

    print(f"检测到新调整事件：{row['title']}（公告日{publish_date.date()}，生效日{eff_date.date()}）")
    print(f"调入{len(added)}只，按流动性精简为{len(targets)}只：{targets}")
    print(f"预计建仓日：{entry_date.date() if entry_date is not None else '无法推算'}（T+1建仓）")
    print(f"预计退出日：{exit_date.date() if exit_date is not None else '无法推算'}（生效日前1个交易日）")
    return log


def handle_active_event(active: pd.Series, log: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    idx = log.index[log["ann_id"].astype(str) == str(active["ann_id"])][-1]
    entry_date = pd.Timestamp(active["entry_date"]) if pd.notna(active["entry_date"]) else None
    exit_date = pd.Timestamp(active["exit_date"]) if pd.notna(active["exit_date"]) else None
    targets = [c for c in str(active["target_codes"]).split(",") if c]

    if active["status"] == STATUS_WAITING_ENTRY:
        if entry_date is not None and today >= entry_date:
            capital_per_stock = CAPITAL_YUAN / len(targets) if targets else 0
            print(f"=== 建仓提醒：请今天以最新价买入以下{len(targets)}只标的 ===")
            for code in targets:
                shares = suggested_shares(code, capital_per_stock)
                print(f"  {code}  建议买入 {shares} 股")
            log.loc[idx, "status"] = STATUS_HOLDING
        else:
            days_left = "未知" if entry_date is None else (entry_date - today).days
            print(f"等待建仓，预计建仓日{entry_date.date() if entry_date is not None else '无法推算'}"
                  f"（还差约{days_left}天）")

    elif active["status"] == STATUS_HOLDING:
        if exit_date is not None and today >= exit_date:
            print(f"=== 退出提醒：请今天卖出全部持仓（生效日前1个交易日到达） ===")
            for code in targets:
                print(f"  {code}  全部卖出")
            log.loc[idx, "status"] = STATUS_DONE
        else:
            days_left = "未知" if exit_date is None else (exit_date - today).days
            print(f"持仓中，预计退出日{exit_date.date() if exit_date is not None else '无法推算'}"
                  f"（还差约{days_left}天）")

    return log


def main():
    pro = init_pro()
    trade_days = load_trade_days(pro)
    today = pd.Timestamp.today().normalize()

    log = load_log()
    active = get_active_row(log)

    if active is None:
        log = register_new_event(pro, trade_days, log)
    else:
        log = handle_active_event(active, log, today)

    save_log(log)


if __name__ == "__main__":
    main()
