"""
ETF 轮动策略 V4 — 新信号叠加验证（2026-07）

基线：风险调整动量 + 拥挤度软过滤（threshold=0.75, factor=0.2），夏普1.005，最大回撤-25.8%

新增信号：
  1. 社融信号（全局时序信号）
     - 当月社融新增相对过去 24 个月历史分位数
     - 强月（分位数 > sf_strong_threshold）：动量得分 × 1.0（不放大，维持满仓）
     - 弱月（分位数 < sf_weak_threshold）：全体 ETF 动量得分 × sf_penalty
     - 注意：不降总仓位，只做横截面重排（高得分=选出来，惩罚不影响具体持仓比例）
     - 理论依据：社融弱月 IC=0.122，IC>0=58.8%（弱有效），强月行业ETF平均+1.76% vs 弱月+0.16%

  2. ETF 资金净流量反向软过滤
     - 上月 ETF 份额净变化率（月末份额变化 / 上月末份额）
     - 净流出 ETF（fund_flow < 0）：动量得分 × flow_boost（>1，提升排名）
     - 净流入 ETF（fund_flow > 0）：动量得分 × flow_penalty（<1，降低排名）
     - 理论依据：IC=-0.062，IC>0=37.5%，Q1(净流出)次月+1.11% vs Q3(净流入)-0.18%

注意事项：
  - 资金流数据从 2019 年起可用；2019 年前回测段自动 fallback（不使用资金流信号）
  - 社融数据全覆盖 2016-2026
  - 回测成本：佣金万1双边 + 滑点万2双边

参数网格：
  - sf_penalty ∈ {0.5, 0.7, 0.9, 1.0（关闭）}
  - flow_boost ∈ {1.0（关闭）, 1.1, 1.2, 1.3}
  - flow_penalty ∈ {0.7, 0.85, 1.0（关闭）}
  - 组合中 sf_penalty=1.0 且 flow_boost=1.0 等价于纯基线
"""

import sys
import time
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro

# ── 参数 ──────────────────────────────────────────────────
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001
SLIPPAGE         = 0.0002
BENCHMARK        = "510300.SH"
START_DATE       = "2016-01-01"
IS_RATIO         = 0.8
MOMENTUM_WINDOW  = 25
TOP_N            = 3
RISK_VOL_WINDOW  = 21
CORR_WINDOW      = 60
CORR_HIST_WINDOW = 252

# 固定基线配置：拥挤度软过滤（已验证最优）
CROWD_THRESHOLD  = 0.75
CROWD_FACTOR     = 0.20

# 社融信号参数网格
SF_WEAK_THRESHOLDS = [0.30, 0.40]     # 低于此分位数视为弱月
SF_PENALTIES       = [0.50, 0.70, 0.90, 1.00]  # 1.0=关闭

# 资金流信号参数网格
FLOW_BOOSTS    = [1.00, 1.10, 1.20, 1.30]   # 净流出ETF放大，1.0=关闭
FLOW_PENALTIES = [0.70, 0.85, 1.00]         # 净流入ETF压制，1.0=关闭

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["font.family"] = ["Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 工具函数 ─────────────────────────────────────────────

def momentum_score(prices):
    y = np.log(prices.values)
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    y_hat = slope * x + np.mean(y - slope * x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope * 252 * r2


def calc_all_scores(close_matrix):
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        ss = pd.Series(index=series.index, dtype=float)
        for i in range(MOMENTUM_WINDOW, len(series)):
            raw = momentum_score(series.iloc[i - MOMENTUM_WINDOW: i])
            if i >= RISK_VOL_WINDOW:
                rets = series.iloc[i - RISK_VOL_WINDOW: i].pct_change().dropna()
                vol  = rets.std() * np.sqrt(252)
                raw  = raw / vol if vol > 1e-6 else raw
            ss.iloc[i] = raw
        scores[code] = ss
    return pd.DataFrame(scores).reindex(close_matrix.index)


def calc_crowding(close):
    """计算行业拥挤度分位数（60日均相关 → 252日历史分位）"""
    codes = [c for c in close.columns if c != BENCHMARK]
    ret   = close[codes].pct_change()
    crowding_raw = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_WINDOW, len(close.index)):
        ret_win = ret.iloc[i - CORR_WINDOW: i].dropna(axis=1, how="any")
        if ret_win.shape[1] < 5:
            continue
        corr_arr = ret_win.corr().values.copy()
        np.fill_diagonal(corr_arr, np.nan)
        avg_corr = pd.Series(np.nanmean(corr_arr, axis=1), index=ret_win.columns)
        crowding_raw.loc[close.index[i], avg_corr.index] = avg_corr.values
    crowding_pct = pd.DataFrame(index=close.index, columns=codes, dtype=float)
    for i in range(CORR_HIST_WINDOW + CORR_WINDOW, len(close.index)):
        date = close.index[i]
        hist = crowding_raw.iloc[i - CORR_HIST_WINDOW: i]
        curr = crowding_raw.iloc[i]
        for code in codes:
            h = hist[code].dropna(); c = curr[code]
            if pd.isna(c) or len(h) < 20:
                crowding_pct.loc[date, code] = np.nan
            else:
                crowding_pct.loc[date, code] = (h < c).mean()
    return crowding_pct


def get_rebalance_dates(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 新信号数据获取 ────────────────────────────────────────

def fetch_sf_score(pro) -> pd.Series:
    """
    获取社融月度数据并构建历史分位数打分（0-1）
    返回：月末日期为索引，值为当月社融新增的过去 24 个月历史分位数
    """
    df = pro.sf_month(start_m="201601", end_m="202606")
    df["date"] = pd.to_datetime(df["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").set_index("date")
    sf_raw = df["inc_month"].astype(float)

    # 滚动24月分位数打分
    sf_score = sf_raw.rolling(24, min_periods=12).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean(), raw=False
    )
    return sf_score.rename("sf_score")


def fetch_fund_share_monthly(pro, codes: list) -> pd.DataFrame:
    """
    批量获取ETF月末份额，返回月度变化率矩阵
    返回：index=月末日期，columns=ETF代码，value=单月份额变化率
    负值=净流出，正值=净流入
    """
    today = pd.Timestamp.today().strftime("%Y%m%d")
    frames = {}
    total = len(codes)
    print(f"  获取 {total} 只ETF份额数据...")
    for i, code in enumerate(codes, 1):
        try:
            df = pro.fund_share(ts_code=code, start_date="20190101", end_date=today)
            if not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").set_index("trade_date")
                frames[code] = df["fd_share"].astype(float)
            time.sleep(0.15)
            if i % 15 == 0:
                print(f"    已获取 {i}/{total}...")
        except Exception as e:
            print(f"    {code} 失败: {e}")

    if not frames:
        return pd.DataFrame()

    share_matrix = pd.DataFrame(frames)
    monthly_share = share_matrix.resample("ME").last()
    flow_1m = monthly_share.pct_change()  # 当月份额变化率，负=净流出
    return flow_1m


# ── 核心回测 ─────────────────────────────────────────────

def run_bt(close, scores, rebal_dates, crowding_pct,
           sf_score=None, sf_weak_thr=0.40, sf_penalty=1.0,
           flow_matrix=None, flow_boost=1.0, flow_penalty=1.0,
           is_baseline=False):
    """
    回测主循环

    Parameters
    ----------
    is_baseline : bool
        True = 纯基线（不使用拥挤度过滤和新信号）
    crowding_pct : DataFrame
        行业拥挤度分位数（固定参数：0.75/0.2）
    sf_score : Series
        社融历史分位数（月末索引），None 则不使用社融信号
    sf_weak_thr : float
        社融弱月阈值，低于此分位数认为社融偏弱
    sf_penalty : float
        社融弱月时全体 ETF 动量得分乘数，<1 降低选股分数（横截面压制）
    flow_matrix : DataFrame
        ETF 月度份额变化率矩阵（月末索引），None 则不使用资金流信号
    flow_boost : float
        净流出 ETF 得分放大系数（>1）
    flow_penalty : float
        净流入 ETF 得分压制系数（<1）
    """
    cash = INIT_CASH
    holdings = {}
    nav_series = pd.Series(index=close.index, dtype=float)
    rebal_set = set(rebal_dates)

    # 将 sf_score 对齐到交易日索引（月末信号，在下月第一个交易日使用）
    sf_monthly = None
    if sf_score is not None:
        sf_monthly = sf_score.copy()

    # 将 flow_matrix 对齐（上月份额变化，在当月第一个交易日使用）
    flow_monthly = None
    if flow_matrix is not None:
        flow_monthly = flow_matrix.copy()

    for date in close.index:
        pv = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                pv += shares * close.loc[date, code]
        nav_series[date] = pv

        if date not in rebal_set:
            continue

        ds = scores.loc[date].dropna().copy()

        # ── 信号1：拥挤度软过滤（固定参数，非基线时启用）
        if not is_baseline and crowding_pct is not None and date in crowding_pct.index:
            dc = crowding_pct.loc[date]
            for code in ds.index:
                if code in dc.index and not pd.isna(dc[code]) and dc[code] > CROWD_THRESHOLD:
                    ds[code] *= CROWD_FACTOR

        # ── 信号2：社融信号（全局时序，弱月横截面压制）
        if sf_monthly is not None and sf_penalty < 1.0:
            # 找当月或上月末的社融分位数
            past_sf = sf_monthly[sf_monthly.index < date]
            if not past_sf.empty:
                last_sf = past_sf.iloc[-1]
                if not pd.isna(last_sf) and last_sf < sf_weak_thr:
                    ds = ds * sf_penalty  # 弱月：全体分数降权，只影响横截面排序

        # ── 信号3：ETF 资金净流量反向软过滤
        if flow_monthly is not None and (flow_boost != 1.0 or flow_penalty != 1.0):
            # 找上月末的资金流数据
            past_flow = flow_monthly[flow_monthly.index < date]
            if not past_flow.empty:
                last_flow = past_flow.iloc[-1]
                for code in ds.index:
                    if code in last_flow.index and not pd.isna(last_flow[code]):
                        fv = last_flow[code]
                        if fv < 0:  # 净流出 → 反向做多
                            ds[code] *= flow_boost
                        elif fv > 0:  # 净流入 → 降低权重
                            ds[code] *= flow_penalty

        tc = list(ds[ds > 0].nlargest(TOP_N).index)

        # 卖出不在持仓列表的标的
        for code in list(holdings.keys()):
            if code not in tc:
                price = close.loc[date, code] if code in close.columns else None
                if price is not None and not pd.isna(price):
                    cash += holdings[code] * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                del holdings[code]

        if not tc:
            continue

        n = len(tc)
        weights = {c: 1.0 / n for c in tc}
        for code in tc:
            price = close.loc[date, code] if code in close.columns else None
            if price is None or pd.isna(price):
                continue
            bp = price * (1 + SLIPPAGE / 2)
            tv = pv * weights[code]
            cs = holdings.get(code, 0)
            cv = cs * price
            diff = tv - cv
            if diff > bp * 100:
                bs = int(diff / bp / 100) * 100
                if bs > 0:
                    cost = bs * bp * (1 + COMMISSION)
                    if cash >= cost:
                        cash -= cost
                        holdings[code] = cs + bs
            elif diff < -price * 100:
                ss = int(-diff / price / 100) * 100
                if ss > 0 and cs >= ss:
                    cash += ss * price * (1 - SLIPPAGE / 2) * (1 - COMMISSION)
                    holdings[code] = cs - ss

    return nav_series.dropna()


def calc_full_stats(nav: pd.Series, label: str = "") -> dict:
    rets   = nav.pct_change().dropna()
    years  = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr   = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino  = cagr / downside if downside > 0 else 0
    monthly  = nav.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    wins     = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    losses   = monthly[monthly < 0].abs().mean() if (monthly < 0).any() else 1
    pnl_ratio = wins / losses if losses > 0 else 0
    return {
        "标的":     label,
        "年化收益": f"{cagr*100:.1f}%",
        "夏普":     f"{sharpe:.3f}",
        "最大回撤": f"{max_dd*100:.1f}%",
        "Calmar":   f"{calmar:.2f}",
        "Sortino":  f"{sortino:.2f}",
        "月胜率":   f"{win_rate:.1%}",
        "盈亏比":   f"{pnl_ratio:.2f}",
        "_sharpe":  sharpe,
        "_maxdd":   max_dd,
        "_cagr":    cagr,
    }


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ETF 轮动 V4 — 社融信号 + ETF 资金净流量反向软过滤 验证")
    print("基线：风险调整动量 + 拥挤度软过滤（0.75, 0.2），夏普1.005")
    print("=" * 70)

    # ── 加载价格数据
    print("\n加载价格数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE].copy()
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，{close.index[0].date()} ~ {close.index[-1].date()}")

    # ── 计算动量得分和拥挤度
    print("计算动量得分...")
    scores = calc_all_scores(close)
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("计算拥挤度分位数...")
    crowding_pct = calc_crowding(close)

    # ── IS/OOS 切分
    n_days    = len(close)
    split_idx = int(n_days * IS_RATIO)
    split_date = close.index[split_idx]
    print(f"\nIS/OOS 分割：{close.index[0].date()} ~ {split_date.date()} | {split_date.date()} ~ {close.index[-1].date()}")

    close_is  = close[close.index <  split_date]
    close_oos = close[close.index >= split_date]
    rebal_is  = [d for d in rebal_dates if d <  split_date]
    rebal_oos = [d for d in rebal_dates if d >= split_date]
    sc_is  = scores[scores.index <  split_date]
    sc_oos = scores[scores.index >= split_date]
    cp_is  = crowding_pct[crowding_pct.index <  split_date]
    cp_oos = crowding_pct[crowding_pct.index >= split_date]

    # ── 获取新信号数据
    print("\n获取社融数据...")
    pro = init_pro()
    sf_score = fetch_sf_score(pro)
    print(f"社融分位数：{len(sf_score)} 个月，{sf_score.index[0].date()} ~ {sf_score.index[-1].date()}")
    sf_is  = sf_score[sf_score.index < split_date]
    sf_oos = sf_score[sf_score.index >= split_date]

    print("\n获取ETF份额数据（资金流信号）...")
    etf_codes = [c for c in valid_codes if c != BENCHMARK]
    flow_matrix = fetch_fund_share_monthly(pro, etf_codes)
    if flow_matrix.empty:
        print("警告：资金流数据为空，将跳过资金流信号测试")
        flow_matrix = None
    else:
        print(f"资金流矩阵：{flow_matrix.shape}，{flow_matrix.index[0].date()} ~ {flow_matrix.index[-1].date()}")
        flow_is  = flow_matrix[flow_matrix.index < split_date] if flow_matrix is not None else None
        flow_oos = flow_matrix[flow_matrix.index >= split_date] if flow_matrix is not None else None

    # ── 基线（纯动量，无拥挤度）
    print("\n运行基线回测（纯动量，无信号）...")
    nav_base_full = run_bt(close,    scores, rebal_dates, crowding_pct,
                           is_baseline=True)
    nav_base_is   = run_bt(close_is, sc_is,  rebal_is,   cp_is,
                           is_baseline=True)
    nav_base_oos  = run_bt(close_oos,sc_oos, rebal_oos,  cp_oos,
                           is_baseline=True)
    s_base = calc_full_stats(nav_base_full, "纯动量基线")
    print(f"  纯动量基线  全样本夏普={s_base['_sharpe']:.3f}")

    # ── 拥挤度基线（V3最优）
    print("运行拥挤度基线（V3最优：0.75, 0.2）...")
    nav_crowd_full = run_bt(close,    scores, rebal_dates, crowding_pct)
    nav_crowd_is   = run_bt(close_is, sc_is,  rebal_is,   cp_is)
    nav_crowd_oos  = run_bt(close_oos,sc_oos, rebal_oos,  cp_oos)
    s_crowd = calc_full_stats(nav_crowd_full, "拥挤度基线(0.75,0.2)")
    print(f"  拥挤度基线  全样本夏普={s_crowd['_sharpe']:.3f}（参考值1.005）")

    # ── 参数网格：社融信号 × 资金流信号
    print("\n" + "=" * 70)
    print("参数网格回测...")
    print("=" * 70)

    # 设计测试矩阵：
    # A. 只加社融信号
    # B. 只加资金流信号
    # C. 同时加两个信号（最佳组合）
    candidates = []

    # A. 社融信号参数扫描（基于拥挤度基线叠加）
    for sf_thr in SF_WEAK_THRESHOLDS:
        for sf_pen in SF_PENALTIES:
            if sf_pen == 1.0:
                continue  # 跳过等价于基线的配置
            label = f"SF(thr={sf_thr:.2f},pen={sf_pen:.2f})"
            candidates.append(("sf_only", label, sf_thr, sf_pen, 1.0, 1.0))

    # B. 资金流信号参数扫描
    if flow_matrix is not None:
        for fb in FLOW_BOOSTS:
            for fp in FLOW_PENALTIES:
                if fb == 1.0 and fp == 1.0:
                    continue
                label = f"Flow(boost={fb:.2f},pen={fp:.2f})"
                candidates.append(("flow_only", label, 0.40, 1.0, fb, fp))

    # C. 最有希望的组合（社融弱惩罚0.7 × 资金流boost1.2/pen0.85）
    if flow_matrix is not None:
        for sf_thr, sf_pen in [(0.40, 0.70), (0.40, 0.50)]:
            for fb, fp in [(1.20, 0.85), (1.10, 1.00), (1.20, 1.00)]:
                label = f"SF({sf_thr:.2f},{sf_pen:.2f})+Flow({fb:.2f},{fp:.2f})"
                candidates.append(("combined", label, sf_thr, sf_pen, fb, fp))

    results = []
    total = len(candidates)
    print(f"共 {total} 个参数组合（+ 2 条基线）\n")

    # 辅助函数：处理IS/OOS的资金流数据分割
    def get_flow_split(fm, split_dt):
        if fm is None:
            return None, None
        return (fm[fm.index < split_dt],
                fm[fm.index >= split_dt])

    flow_is, flow_oos = get_flow_split(flow_matrix, split_date)

    for i, (group, label, sf_thr, sf_pen, fb, fp) in enumerate(candidates, 1):
        # 全样本
        fm = flow_matrix if group in ("flow_only", "combined") else None
        sfm = sf_score if group in ("sf_only", "combined") else None

        nav_full = run_bt(close,    scores, rebal_dates, crowding_pct,
                          sf_score=sfm, sf_weak_thr=sf_thr, sf_penalty=sf_pen,
                          flow_matrix=fm, flow_boost=fb, flow_penalty=fp)
        # IS
        fm_i  = flow_is  if group in ("flow_only", "combined") else None
        sfm_i = sf_is    if group in ("sf_only", "combined") else None
        nav_is_bt = run_bt(close_is, sc_is, rebal_is, cp_is,
                           sf_score=sfm_i, sf_weak_thr=sf_thr, sf_penalty=sf_pen,
                           flow_matrix=fm_i, flow_boost=fb, flow_penalty=fp)
        # OOS
        fm_o  = flow_oos  if group in ("flow_only", "combined") else None
        sfm_o = sf_oos    if group in ("sf_only", "combined") else None
        nav_oos_bt = run_bt(close_oos, sc_oos, rebal_oos, cp_oos,
                            sf_score=sfm_o, sf_weak_thr=sf_thr, sf_penalty=sf_pen,
                            flow_matrix=fm_o, flow_boost=fb, flow_penalty=fp)

        sf = calc_full_stats(nav_full)
        si = calc_full_stats(nav_is_bt)
        so = calc_full_stats(nav_oos_bt)
        decay = so["_sharpe"] / si["_sharpe"] if si["_sharpe"] > 0 else 0
        delta_crowd = sf["_sharpe"] - s_crowd["_sharpe"]

        results.append({
            "组": group,
            "配置": label,
            "全样本夏普": sf["_sharpe"],
            "全样本年化": sf["_cagr"],
            "全样本回撤": sf["_maxdd"],
            "IS夏普": si["_sharpe"],
            "OOS夏普": so["_sharpe"],
            "OOS/IS": decay,
            "vs拥挤度基线": delta_crowd,
        })

        if i % 5 == 0 or i == total:
            print(f"  [{i:2d}/{total}] {label:<45} 全样本={sf['_sharpe']:.3f}  IS={si['_sharpe']:.3f}  OOS={so['_sharpe']:.3f}  Δ={delta_crowd:+.3f}")

    # ── 汇总输出
    print("\n" + "=" * 70)
    print("结果汇总：全样本夏普 vs 拥挤度基线（1.005）")
    print("=" * 70)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("全样本夏普", ascending=False)

    # 添加基线行（便于对比）
    base_row = pd.DataFrame([{
        "组": "baseline",
        "配置": "纯动量基线",
        "全样本夏普": s_base["_sharpe"],
        "全样本年化": s_base["_cagr"],
        "全样本回撤": s_base["_maxdd"],
        "IS夏普": calc_full_stats(nav_base_is)["_sharpe"],
        "OOS夏普": calc_full_stats(nav_base_oos)["_sharpe"],
        "OOS/IS": calc_full_stats(nav_base_oos)["_sharpe"] / calc_full_stats(nav_base_is)["_sharpe"],
        "vs拥挤度基线": s_base["_sharpe"] - s_crowd["_sharpe"],
    }])
    crowd_row = pd.DataFrame([{
        "组": "baseline",
        "配置": "拥挤度基线(0.75,0.2)",
        "全样本夏普": s_crowd["_sharpe"],
        "全样本年化": s_crowd["_cagr"],
        "全样本回撤": s_crowd["_maxdd"],
        "IS夏普": calc_full_stats(nav_crowd_is)["_sharpe"],
        "OOS夏普": calc_full_stats(nav_crowd_oos)["_sharpe"],
        "OOS/IS": calc_full_stats(nav_crowd_oos)["_sharpe"] / calc_full_stats(nav_crowd_is)["_sharpe"],
        "vs拥挤度基线": 0.0,
    }])
    df_all = pd.concat([crowd_row, base_row, df_res], ignore_index=True)

    # 格式化输出
    print(f"\n{'配置':<48} {'全样本夏普':>8} {'年化':>7} {'回撤':>7} {'IS夏普':>7} {'OOS夏普':>8} {'OOS/IS':>7} {'Δ拥挤度':>8}")
    print("-" * 105)
    for _, row in df_all.iterrows():
        marker = "★" if row["vs拥挤度基线"] > 0.03 else ("△" if row["vs拥挤度基线"] > 0 else " ")
        print(f"{row['配置']:<48} {row['全样本夏普']:>8.3f} {row['全样本年化']*100:>6.1f}% {row['全样本回撤']*100:>6.1f}% {row['IS夏普']:>7.3f} {row['OOS夏普']:>8.3f} {row['OOS/IS']:>7.2f} {row['vs拥挤度基线']:>+8.3f} {marker}")

    # ── 分组统计：社融 vs 资金流 vs 组合
    print("\n" + "=" * 70)
    print("分组有效性统计")
    print("=" * 70)
    for group_name in ["sf_only", "flow_only", "combined"]:
        sub = df_res[df_res["组"] == group_name]
        if sub.empty:
            continue
        n_better = (sub["vs拥挤度基线"] > 0).sum()
        n_total  = len(sub)
        print(f"\n[{group_name}] {n_total} 个配置：")
        print(f"  超过拥挤度基线（夏普）比例：{n_better}/{n_total} = {n_better/n_total:.0%}")
        print(f"  全样本夏普均值={sub['全样本夏普'].mean():.3f}  最优={sub['全样本夏普'].max():.3f}  最差={sub['全样本夏普'].min():.3f}")
        best = sub.loc[sub["全样本夏普"].idxmax()]
        print(f"  最优配置：{best['配置']}，IS={best['IS夏普']:.3f}，OOS={best['OOS夏普']:.3f}，OOS/IS={best['OOS/IS']:.2f}")

    # ── 多重测试评估
    n_trials = len(results)
    n_better = (df_res["vs拥挤度基线"] > 0).sum()
    print(f"\n多重测试：测试了 {n_trials} 个参数组合，超过拥挤度基线的比例：{n_better}/{n_trials} = {n_better/n_trials:.0%}")
    if n_better / n_trials > 0.55:
        print("→ >55% 参数超过基线，信号具有系统性效果，不是参数选择的孤立尖峰")
    elif n_better / n_trials > 0.35:
        print("→ 35-55% 参数超过基线，信号效果有限，需谨慎")
    else:
        print("→ <35% 参数超过基线，信号无效，建议放弃")

    # ── 可视化：净值曲线对比（取各组最优配置）
    print("\n生成图表...")

    # 找各组最优
    best_configs = {}
    for group_name in ["sf_only", "flow_only", "combined"]:
        sub = df_res[df_res["组"] == group_name]
        if not sub.empty:
            best_configs[group_name] = sub.loc[sub["全样本夏普"].idxmax()]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 净值曲线
    ax1 = axes[0]
    ax1.plot(nav_base_full.index, nav_base_full / INIT_CASH,
             color="#9E9E9E", lw=1.5, linestyle="--", label=f"纯动量基线 夏普={s_base['_sharpe']:.3f}")
    ax1.plot(nav_crowd_full.index, nav_crowd_full / INIT_CASH,
             color="#1565C0", lw=2.0, label=f"拥挤度基线(0.75,0.2) 夏普={s_crowd['_sharpe']:.3f}")

    colors_extra = {"sf_only": "#E53935", "flow_only": "#43A047", "combined": "#7B1FA2"}

    # 将 candidates 列表转为字典，方便按配置名查找参数
    cand_params = {label: (group, sf_thr, sf_pen, fb, fp)
                   for group, label, sf_thr, sf_pen, fb, fp in candidates}

    for group_name, best in best_configs.items():
        # 直接从 candidates 中取参数，避免字符串解析
        best_label = best["配置"]
        if best_label not in cand_params:
            continue
        _, sf_thr_val, sf_pen_val, fb_val, fp_val = cand_params[best_label]

        sfm = sf_score if group_name in ("sf_only", "combined") else None
        fm  = flow_matrix if group_name in ("flow_only", "combined") else None

        nav_best = run_bt(close, scores, rebal_dates, crowding_pct,
                          sf_score=sfm, sf_weak_thr=sf_thr_val, sf_penalty=sf_pen_val,
                          flow_matrix=fm, flow_boost=fb_val, flow_penalty=fp_val)

        ax1.plot(nav_best.index, nav_best / INIT_CASH,
                 color=colors_extra[group_name], lw=1.6,
                 label=f"{best['配置'][:40]} 夏普={best['全样本夏普']:.3f}")

    ax1.axvline(split_date, color="red", linestyle="--", alpha=0.4, lw=1)
    ax1.set_title("ETF轮动 V4 — 新信号叠加净值对比（2016-2026）")
    ax1.set_ylabel("净值")
    ax1.legend(fontsize=7, ncol=1)
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.3)

    # 柱状图：各参数组合全样本夏普
    ax2 = axes[1]
    df_plot = df_res.copy().head(20)  # 只展示夏普前20
    colors_bar = [colors_extra.get(g, "#888") for g in df_plot["组"]]
    bars = ax2.barh(range(len(df_plot)), df_plot["全样本夏普"], color=colors_bar, alpha=0.7)
    ax2.axvline(s_crowd["_sharpe"], color="#1565C0", linestyle="--", lw=1.5,
                label=f"拥挤度基线={s_crowd['_sharpe']:.3f}")
    ax2.axvline(s_base["_sharpe"], color="#9E9E9E", linestyle="--", lw=1.0,
                label=f"纯动量={s_base['_sharpe']:.3f}")
    ax2.set_yticks(range(len(df_plot)))
    ax2.set_yticklabels([c[:40] for c in df_plot["配置"]], fontsize=6)
    ax2.set_title("参数网格全样本夏普（前20名，红=社融，绿=资金流，紫=组合）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = RESULTS_DIR / "etf_rotation_v4_newfilters.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{out_path}")

    # ── 结论
    print("\n" + "=" * 70)
    print("验收结论")
    print("=" * 70)
    print(f"拥挤度基线：夏普={s_crowd['_sharpe']:.3f}，年化={s_crowd['年化收益']}，最大回撤={s_crowd['最大回撤']}")
    print()

    for group_name, group_label in [("sf_only", "社融信号"), ("flow_only", "资金流反向信号"), ("combined", "组合信号")]:
        sub = df_res[df_res["组"] == group_name]
        if sub.empty:
            print(f"{group_label}：无数据（可能资金流获取失败）")
            continue
        best = sub.loc[sub["全样本夏普"].idxmax()]
        n_better = (sub["vs拥挤度基线"] > 0).sum()
        delta = best["全样本夏普"] - s_crowd["_sharpe"]
        oos_is = best["OOS/IS"]

        if delta > 0.05 and oos_is >= 0.5:
            verdict = "有效 ★ 建议纳入策略"
        elif delta > 0 and oos_is >= 0.4:
            verdict = "边际正向，需更多验证"
        elif delta > 0:
            verdict = f"全样本正向但OOS/IS={oos_is:.2f}，疑似过拟合"
        else:
            verdict = "无效，放弃"

        print(f"{group_label}：")
        print(f"  最优配置：{best['配置']}")
        print(f"  全样本夏普={best['全样本夏普']:.3f}（Δ={delta:+.3f}），IS={best['IS夏普']:.3f}，OOS={best['OOS夏普']:.3f}，OOS/IS={oos_is:.2f}")
        print(f"  超基线参数比例：{n_better}/{len(sub)} = {n_better/len(sub):.0%}")
        print(f"  → {verdict}")
        print()


if __name__ == "__main__":
    main()
