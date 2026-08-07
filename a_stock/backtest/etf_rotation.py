"""
ETF 趋势轮动回测
策略：OLS动量评分（斜率×R²）月度调仓，持仓 Top N，全现金空仓保护
成本：佣金万1双边 + 滑点万2双边（ETF无印花税）
基准：沪深300（510300.SH）买入持有

优化选项：
  cash_etf               - 空仓时停泊于货币ETF（银华日利511880），而非持现金
  use_risk_adj_momentum  - 信号除以近期波动率（风险调整动量），降低高波动时期权重
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 数据目录
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix
from etf_universe import ETF_UNIVERSE

# ── 参数 ─────────────────────────────────────────────────
MOMENTUM_WINDOW  = 25       # OLS动量窗口（交易日）
TOP_N            = 3        # 持仓只数
INIT_CASH        = 1_000_000
COMMISSION       = 0.0001   # 佣金万1（单边），双边合计万2
SLIPPAGE         = 0.0002   # 滑点万2（双边合计），买卖各万1
BENCHMARK        = "510300.SH"
START_DATE       = "2016-01-01"  # 留足动量预热期
MARKET_FILTER_MA     = 200      # 大盘趋势过滤均线周期（交易日）
IVOL_WINDOW          = 20       # 波动率反比加权计算窗口（交易日）
TRAILING_STOP_PCT   = 0.20   # 追踪止损阈值：持仓最高点回撤超过20%触发
COOLDOWN_DAYS       = 15     # 止损后冷却天数
CORR_THRESHOLD      = 0.70   # 相关性过滤阈值
CORR_WINDOW         = 60     # 相关性计算窗口（交易日）
CASH_ETF             = "511880.SH"  # 空仓停泊货币ETF（银华日利），None 则持现金
RISK_ADJ_VOL_WINDOW  = 21    # 风险调整动量：除以近N日年化波动率

# ── 动量评分：OLS斜率 × R² ───────────────────────────────

def momentum_score(prices: pd.Series) -> float:
    """
    对过去 N 日对数收益率做 OLS，计算年化斜率 × R²。
    斜率：每日对数收益趋势，乘以252年化。
    R²：趋势稳定性，过滤噪声。
    得分 > 0 表示上升趋势，否则为空。
    """
    y = np.log(prices.values)
    x = np.arange(len(y))
    # OLS
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    annualized_slope = slope * 252
    return annualized_slope * r2


def calc_all_scores(
    close_matrix: pd.DataFrame,
    window: int,
    risk_adj: bool = False,
    risk_vol_window: int = RISK_ADJ_VOL_WINDOW,
) -> pd.DataFrame:
    """
    计算所有标的每个交易日的动量得分，返回与 close_matrix 同索引的 DataFrame。
    risk_adj=True 时，得分除以近 risk_vol_window 日年化波动率，
    自动降低高波动时期的排名权重（风险调整动量）。
    """
    scores = {}
    for code in close_matrix.columns:
        series = close_matrix[code].dropna()
        score_series = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            price_window = series.iloc[i - window: i]
            raw_score = momentum_score(price_window)
            if risk_adj and i >= risk_vol_window:
                # 用动量窗口起点前 risk_vol_window 日的收益率计算波动率
                ret_window = series.iloc[i - risk_vol_window: i].pct_change().dropna()
                vol = ret_window.std() * np.sqrt(252)
                raw_score = raw_score / vol if vol > 1e-6 else raw_score
            score_series.iloc[i] = raw_score
        scores[code] = score_series
    return pd.DataFrame(scores).reindex(close_matrix.index)


# ── 月度调仓日 ────────────────────────────────────────────

def get_rebalance_dates(index: pd.DatetimeIndex) -> list:
    """每月第一个交易日"""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return df.groupby("ym").apply(lambda x: x.index[0]).tolist()


# ── 回测主逻辑 ────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION,
    slippage: float = SLIPPAGE,
    use_market_filter: bool = False,
    use_ivol_weighting: bool = False,
    use_trailing_stop: bool = False,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    cooldown_days: int = COOLDOWN_DAYS,
    use_corr_filter: bool = False,
    corr_threshold: float = CORR_THRESHOLD,
    corr_window: int = CORR_WINDOW,
    use_industry_cap: bool = False,
    industry_map: dict = None,
    max_per_industry: int = 1,
    cash_etf: str = None,
    amount_wide: pd.DataFrame = None,
    impact_coef: float = 0.0,
) -> pd.DataFrame:
    """
    逐月调仓模拟，返回每日资产净值序列。
    执行价格：调仓日收盘价（保守，实盘用次日开盘，差异小）。
    T+1：买入次日才能卖出，月度调仓不触发T+1约束。

    use_industry_cap：行业分散约束，同一行业（industry_map 给出的分类）最多入选
    max_per_industry 只，贪心法按得分高低逐一加入，超限则跳过换下一个候选。

    amount_wide/impact_coef：可选的成交量冲击成本模型（简化线性模型，非
    Almgren-Chriss平方根模型，仅用于粗略敏感性测试）。amount_wide 为
    index=trade_date, columns=ts_code 的当日成交额宽表（单位：千元，与
    market_turnover.parquet 的 amount 字段一致）。impact_coef=0（默认）时
    完全不影响现有调用与历史结论。>0 时，额外单边冲击成本 =
    impact_coef * (交易金额 / 当日成交额)，叠加在 slippage 之上。
    """
    def _impact(code: str, date, trade_value: float) -> float:
        if impact_coef <= 0 or amount_wide is None or code not in amount_wide.columns:
            return 0.0
        day_amount_qian = amount_wide.loc[date, code] if date in amount_wide.index else np.nan
        if pd.isna(day_amount_qian) or day_amount_qian <= 0:
            return 0.0
        day_amount_yuan = day_amount_qian * 1000
        participation = trade_value / day_amount_yuan
        return impact_coef * participation
    # 预计算大盘 MA200（用于趋势过滤）
    if use_market_filter and BENCHMARK in close.columns:
        benchmark_ma200 = close[BENCHMARK].rolling(MARKET_FILTER_MA).mean()
    else:
        benchmark_ma200 = None
    cash = init_cash
    entry_high = {}     # {ts_code: float} 持仓最高价（买入后追踪）
    cooling_down = {}   # {ts_code: pd.Timestamp} 被止损标的的解禁日期
    holdings = {}   # {ts_code: shares}
    nav_series = pd.Series(index=close.index, dtype=float)

    rebal_set = set(rebal_dates)
    prev_date = None

    for date in close.index:
        # 计算当日净值
        port_value = cash
        for code, shares in holdings.items():
            if code in close.columns and not pd.isna(close.loc[date, code]):
                port_value += shares * close.loc[date, code]
        nav_series[date] = port_value

        # 追踪止损：每日检查持仓最高点回撤
        if use_trailing_stop and holdings:
            for code in list(holdings.keys()):
                price_today = close.loc[date, code] if code in close.columns else None
                if price_today is None or pd.isna(price_today):
                    continue
                # 更新持仓期内最高价
                entry_high[code] = max(entry_high.get(code, price_today), price_today)
                # 检查回撤（负数，绝对值为回撤幅度）
                drawdown = (price_today - entry_high[code]) / entry_high[code]
                if drawdown < -trailing_stop_pct:
                    sell_price = price_today * (1 - slippage / 2)
                    cash += holdings[code] * sell_price * (1 - commission)
                    cooling_down[code] = date + pd.Timedelta(days=cooldown_days)
                    del holdings[code]
                    del entry_high[code]

        # 调仓日
        if date in rebal_set:
            # 大盘趋势过滤：若沪深300 < MA200，清仓保持空仓
            if use_market_filter and benchmark_ma200 is not None:
                ma200_val = benchmark_ma200.get(date)
                bench_close = close[BENCHMARK].get(date) if BENCHMARK in close.columns else None
                # MA200 预热期内（前200日）rolling mean 为 NaN，视为趋势不满足（保守处理）
                market_in_trend = (
                    pd.notna(bench_close)
                    and pd.notna(ma200_val)
                    and bench_close > ma200_val
                )
            else:
                market_in_trend = True  # 不过滤时默认允许建仓

            if not market_in_trend:
                # 大盘趋势不满足，清空全部持仓
                for code in list(holdings.keys()):
                    price = close.loc[date, code] if code in close.columns else None
                    if price is not None and not pd.isna(price):
                        sell_price = price * (1 - slippage / 2)
                        proceeds = holdings[code] * sell_price * (1 - commission)
                        cash += proceeds
                    del holdings[code]
                    if code in entry_high:
                        del entry_high[code]
                continue  # 跳过本月建仓

            # 获取当日有效得分（用当日收盘前的信号，即当日分数已计算完毕）
            day_scores = scores.loc[date].dropna()
            # 候选集扩充（为相关性过滤/行业分散约束留出备选）
            need_expand = use_corr_filter or use_industry_cap
            candidate_size = top_n * 3 if need_expand else top_n
            pos_scores = day_scores[day_scores > 0].nlargest(candidate_size)
            candidates = list(pos_scores.index)

            # 冷却期过滤：排除被止损且仍在冷却中的标的
            if use_trailing_stop:
                candidates = [c for c in candidates
                              if c not in cooling_down or cooling_down[c] <= date]

            # 相关性过滤（贪心法，按得分高低逐一加入，排除高相关标的）
            if use_corr_filter and len(candidates) > 1:
                date_loc = close.index.get_loc(date)
                window_start = max(0, date_loc - corr_window)
                # 用日收益率计算相关性（避免价格水平带来的虚假相关）
                ret_window = close.iloc[window_start:date_loc].pct_change().dropna()
                selected = []
                for code in candidates:
                    if len(selected) >= top_n:
                        break
                    if len(selected) == 0:
                        selected.append(code)
                        continue
                    ok = True
                    for s in selected:
                        pair = ret_window[[code, s]].dropna()
                        if len(pair) < corr_window // 2:
                            continue  # 数据不足，视为不相关，允许加入
                        if pair[code].corr(pair[s]) > corr_threshold:
                            ok = False
                            break
                    if ok:
                        selected.append(code)
                candidates = selected  # 相关性过滤后的候选，若启用行业分散约束将在其上继续贪心

            # 行业分散约束（贪心法，按得分高低逐一加入，同行业超过 max_per_industry 只则跳过）
            if use_industry_cap and industry_map:
                selected = []
                industry_count = {}
                for code in candidates:
                    if len(selected) >= top_n:
                        break
                    ind = industry_map.get(code)
                    if ind is not None and industry_count.get(ind, 0) >= max_per_industry:
                        continue  # 该行业名额已满，跳过换下一个候选
                    selected.append(code)
                    if ind is not None:
                        industry_count[ind] = industry_count.get(ind, 0) + 1
                target_codes = selected
            else:
                target_codes = candidates[:top_n]

            # 先卖出不在目标中的持仓
            for code in list(holdings.keys()):
                if code not in target_codes:
                    price = close.loc[date, code] if code in close.columns else None
                    if price is not None and not pd.isna(price):
                        trade_value = holdings[code] * price
                        impact = _impact(code, date, trade_value)
                        sell_price = price * (1 - slippage / 2 - impact)
                        proceeds = holdings[code] * sell_price * (1 - commission)
                        cash += proceeds
                    del holdings[code]
                    if code in entry_high:
                        del entry_high[code]

            if not target_codes:
                # 无正动量标的：停泊到货币ETF或持现金
                if cash_etf and cash_etf in close.columns:
                    price = close.loc[date, cash_etf]
                    if pd.notna(price) and cash > price:
                        buy_price = price * (1 + slippage / 2)
                        buy_shares = int(cash / buy_price / 100) * 100
                        if buy_shares > 0:
                            cost = buy_shares * buy_price * (1 + commission)
                            if cash >= cost:
                                cash -= cost
                                holdings[cash_etf] = holdings.get(cash_etf, 0) + buy_shares
                continue

            # 仓位分配（等权 or 波动率反比加权）
            # 注意：port_value 是调仓前净值（含当日收盘价），用于定目标仓位；实际买入受 cash 余额约束
            n = len(target_codes)
            if use_ivol_weighting and n > 0:
                # 计算各标的过去 IVOL_WINDOW 日年化波动率
                vols = {}
                for code in target_codes:
                    series = close[code].dropna()
                    loc = series.index.get_loc(date) if date in series.index else -1
                    if loc >= IVOL_WINDOW:
                        ret = series.iloc[loc - IVOL_WINDOW: loc].pct_change().dropna()
                        vol = ret.std() * np.sqrt(252)
                        vols[code] = vol if vol > 0 else 1e-6
                    else:
                        vols[code] = None  # 数据不足，后续用中位数填充
                # 用有效波动率中位数填充数据不足的标的（避免 1e-6 导致权重极度偏斜）
                valid_vols = [v for v in vols.values() if v is not None]
                fallback_vol = float(np.median(valid_vols)) if valid_vols else 1.0
                vols = {c: (v if v is not None else fallback_vol) for c, v in vols.items()}
                inv_vols = {c: 1.0 / v for c, v in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {c: inv_vols[c] / total_inv for c in target_codes}
            else:
                weights = {c: 1.0 / n for c in target_codes}

            for code in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is None or pd.isna(price):
                    continue
                target_value = port_value * weights[code]
                # 已有持仓先计算现有市值
                current_shares = holdings.get(code, 0)
                current_value = current_shares * price
                diff_value = target_value - current_value

                if diff_value > price * 1:  # 至少买1份
                    impact = _impact(code, date, abs(diff_value))
                    buy_price = price * (1 + slippage / 2 + impact)
                    buy_shares = int(diff_value / buy_price / 100) * 100  # ETF按100份整手
                    if buy_shares > 0:
                        cost = buy_shares * buy_price * (1 + commission)
                        if cash >= cost:
                            cash -= cost
                            holdings[code] = current_shares + buy_shares
                            # 新买入标的初始化追踪止损最高价
                            if use_trailing_stop and code not in entry_high:
                                entry_high[code] = price
                elif diff_value < -price * 100:  # 需要减仓
                    sell_shares = int(-diff_value / price / 100) * 100
                    if sell_shares > 0 and current_shares >= sell_shares:
                        impact = _impact(code, date, abs(diff_value))
                        sell_price = price * (1 - slippage / 2 - impact)
                        proceeds = sell_shares * sell_price * (1 - commission)
                        cash += proceeds
                        holdings[code] = current_shares - sell_shares

    return nav_series.dropna()


# ── 业绩指标 ──────────────────────────────────────────────

def calc_stats(nav: pd.Series, label: str = "策略") -> dict:
    returns = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    years = total_days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    # 最大回撤
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    # 年化波动率
    vol = returns.std() * np.sqrt(252)
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "标的": label,
        "总收益": f"{(nav.iloc[-1] / nav.iloc[0] - 1) * 100:.1f}%",
        "年化收益(CAGR)": f"{cagr * 100:.1f}%",
        "年化夏普": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd * 100:.1f}%",
        "年化波动率": f"{vol * 100:.1f}%",
        "Calmar": f"{calmar:.2f}",
    }


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("加载数据...")
    close_full = load_close_matrix()
    close = close_full[close_full.index >= START_DATE]

    # 只保留精选池中有足够历史的标的（上市满1年以上）
    min_records = MOMENTUM_WINDOW + 20
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= min_records]
    close = close[valid_codes]
    print(f"有效标的数：{len(valid_codes)}")

    print(f"计算动量得分（窗口={MOMENTUM_WINDOW}日，共{len(valid_codes)}只）...")
    scores = calc_all_scores(close, MOMENTUM_WINDOW, risk_adj=True)

    rebal_dates = get_rebalance_dates(close.index)
    rebal_dates = [d for d in rebal_dates if d >= pd.Timestamp(START_DATE)]
    print(f"调仓日数量：{len(rebal_dates)}")

    print("运行回测...")
    nav = run_backtest(close, scores, rebal_dates)

    # 基准：沪深300买入持有
    bench = close[BENCHMARK].dropna()
    bench = bench[bench.index >= nav.index[0]]
    bench_nav = bench / bench.iloc[0] * INIT_CASH

    # 对齐日期
    common = nav.index.intersection(bench_nav.index)
    nav = nav[common]
    bench_nav = bench_nav[common]

    # ── 输出结果 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"回测区间：{nav.index[0].date()} → {nav.index[-1].date()}")
    print(f"参数：动量窗口={MOMENTUM_WINDOW}日，Top{TOP_N}持仓，月度调仓\n")

    stats_strategy = calc_stats(nav, f"ETF轮动(Top{TOP_N})")
    stats_bench = calc_stats(bench_nav, "沪深300(买持)")

    result_df = pd.DataFrame([stats_strategy, stats_bench]).set_index("标的")
    print(result_df.to_string())

    # ── 绘图 ─────────────────────────────────────────────
    matplotlib.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'Songti SC', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

    # 净值曲线
    ax1 = axes[0]
    ax1.plot(nav.index, nav / INIT_CASH, label=f"ETF轮动 Top{TOP_N}", color="#2196F3", linewidth=1.5)
    ax1.plot(bench_nav.index, bench_nav / INIT_CASH, label="沪深300买持", color="#FF9800", linewidth=1.2, alpha=0.8)
    ax1.set_title(f"ETF趋势轮动 vs 沪深300（{nav.index[0].year}–{nav.index[-1].year}）")
    ax1.set_ylabel("净值")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    # 回撤
    ax2 = axes[1]
    strategy_dd = (nav - nav.cummax()) / nav.cummax() * 100
    bench_dd = (bench_nav - bench_nav.cummax()) / bench_nav.cummax() * 100
    ax2.fill_between(strategy_dd.index, strategy_dd, 0, alpha=0.4, color="#2196F3", label="策略回撤")
    ax2.fill_between(bench_dd.index, bench_dd, 0, alpha=0.3, color="#FF9800", label="基准回撤")
    ax2.set_ylabel("回撤(%)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / "etf_rotation_result.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存：{fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
