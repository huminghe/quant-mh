"""
可转债双低策略作为分散仓位工具的组合层面测试（方向B，2026-09-18）

背景：research_convertible_bond.md已证伪双低因子作为独立alpha信号（Rank IC未过筛、
信用过滤无效、样本外过拟合等五项证据）。用户决策"往这两个方向投入看看"的第二个方向：
不追求双低值本身的alpha，只测试它加入现有ETF轮动组合后，能否降低组合整体波动率/回撤
（分散工具用法，逻辑类似"加入一个低相关资产降低组合风险"，不要求该资产自身收益领先）。

方法：
1. 复现431池+flow单独配置的ETF轮动月度收益序列（当前线上正式配置，见research_etf_rotation.md）
2. 复用cb_lowprice_backtest.py已产出的可转债双低月度收益序列（无信用过滤版，Top20%分组）
3. 对齐月度日期后，测试不同权重组合（0%-100%逐10%扫描）的年化收益/夏普/最大回撤/
   月度收益相关系数，绝对收益和超额收益（剔除转债市场beta）两个口径都算
4. 滚动24个月窗口重算权重扫描（关键检验）：看全样本权重扫描的峰值是否对样本区间稳健

结论：证伪，不接入分散配置。全样本权重扫描看似正向（80%双低权重夏普从0.644升到
1.124），但滚动窗口检验显示所有测试权重跑输纯ETF轮动的窗口占比32.5%-46.2%，全部
超过trading-standards.md的30%不稳健阈值——全样本正向结论是被少数窗口主导的统计
假象。完整数据表格见 a_stock/docs/research_convertible_bond.md Phase 7。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from etf_rotation import calc_all_scores, get_rebalance_dates, MOMENTUM_WINDOW, TOP_N  # noqa: E402
from etf_rotation_v23_universe_bias_test import CACHE_DIR, load_close_matrix_from_cache  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all  # noqa: E402
from etf_rotation_v18_signal_ablation import run_backtest  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results" / "etf_cb_diversification_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CB_MONTHLY_RET_FILE = (
    pathlib.Path(__file__).parent / "results" / "cb_lowprice_backtest" / "monthly_ret_无过滤.csv"
)
EXPOSURE_FILE = DATA_DIR / "etf_all_candidates.parquet"
START_DATE = "2018-01-01"  # 与可转债回测起点(2018-01)对齐


def build_etf_flow_monthly_returns() -> pd.Series:
    """复现431池+flow单独配置（当前线上正式配置）的月度收益序列。"""
    print("加载431池价格缓存...")
    all_candidates = pd.read_parquet(EXPOSURE_FILE)["ts_code"].tolist()
    close_full = load_close_matrix_from_cache(all_candidates)
    close = close_full[close_full.index >= START_DATE]
    valid_codes = [c for c in close.columns if close[c].notna().sum() >= MOMENTUM_WINDOW + 20]
    close = close[valid_codes]
    print(f"有效标的：{len(valid_codes)} 只，区间：{close.index[0].date()} ~ {close.index[-1].date()}")

    print("计算风险调整动量得分...")
    scores = calc_all_scores(close_full, MOMENTUM_WINDOW, risk_adj=True)[valid_codes]
    scores = scores[scores.index >= START_DATE]
    rebal_dates = [d for d in get_rebalance_dates(close.index) if d >= pd.Timestamp(START_DATE)]

    print("拉取ETF份额数据（flow信号，431只逐一调用tushare，预计较慢）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=START_DATE.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()

    signal_ranks = {}
    for d in rebal_dates:
        flow_d = pd.Series(dtype=float)
        if not flow_1m.empty:
            idx = flow_1m.index[flow_1m.index <= d]
            if len(idx) > 0:
                flow_d = flow_1m.loc[idx[-1]]
        s = flow_d.dropna()
        if len(s) < 5:
            continue
        r = 1 - s.rank(pct=True)  # flow反向：份额流出（低分位）应被打折降权，故取1-rank作boost
        signal_ranks[d] = r

    boost = pd.DataFrame(signal_ranks).T

    print("跑回测（431池+flow单独，当前线上正式配置）...")
    nav = run_backtest(close, scores, rebal_dates, boost_signal=boost, boost_mode="continuous", top_n=TOP_N)
    monthly_nav = nav.resample("ME").last()
    monthly_ret = monthly_nav.pct_change().dropna()
    monthly_ret.index = monthly_ret.index.to_period("M").to_timestamp("M")
    return monthly_ret


def load_cb_monthly_returns(use_excess: bool = False) -> pd.Series:
    """
    cb_lowprice_backtest.py的date列标的是持有期起点（月末建仓日），
    实际收益发生在下一个自然月（建仓到下月末卖出）。ETF轮动序列的月度
    收益索引标的是持有期终点（标准pct_change()语义），两者date含义不同，
    直接按date对齐会错位一个月。这里把cb的date统一shift到收益实际发生
    的月份（即下一个月末），才能和ETF序列在同一个自然月对齐。

    use_excess=True 时返回 strategy - benchmark（剔除转债市场beta后的
    超额收益，口径同research_convertible_bond.md Phase 4b）。绝对收益
    (strategy列)大部分是转债市场beta而非选股alpha（Phase 4b已证实），
    直接用绝对收益做分散测试会把"双低策略收益虚高"的已知问题带回来，
    必须同时看超额收益版本才能诚实回答"双低值能否分散风险"这个问题。
    """
    df = pd.read_csv(CB_MONTHLY_RET_FILE, parse_dates=["date"])
    col = (df["strategy"] - df["benchmark"]) if use_excess else df["strategy"]
    s = col.copy()
    s.index = df["date"]
    # 注意：不能用 date + pd.offsets.MonthEnd(1)——对非月末日期（如2018-03-30）
    # MonthEnd(1)只滚动到"当月"月末（03-31），不是下个月，会产生错误的重复索引。
    # 正确做法：先转Period("M")再+1个自然月，再转回月末时间戳。
    s.index = (pd.to_datetime(s.index).to_period("M") + 1).to_timestamp("M")
    return s


def sharpe(ret: pd.Series, freq: int = 12, rf: float = 0.02) -> float:
    if ret.std() == 0 or len(ret) < 2:
        return 0.0
    excess = ret - rf / freq
    return excess.mean() / ret.std() * np.sqrt(freq)


def max_drawdown_from_ret(ret: pd.Series) -> float:
    nav = (1 + ret).cumprod()
    return ((nav - nav.cummax()) / nav.cummax()).min()


def annual_return(ret: pd.Series, freq: int = 12) -> float:
    nav = (1 + ret).cumprod()
    years = len(ret) / freq
    return nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0


ROLLING_WINDOW_MONTHS = 24


def rolling_diversification_check(
    etf_ret: pd.Series, cb_ret: pd.Series, weights: list[float], label: str
) -> pd.DataFrame:
    """
    权重扫描的峰值可能是全样本单一路径的运气（同cb_lowprice_backtest.py的
    Phase 4逻辑），滚动24个月窗口逐月滑动重算夏普，看"加双低仓位跑赢纯ETF轮动"
    这个结论是否对样本窗口稳健，还是只在特定历史区间成立。
    """
    common_idx = etf_ret.index.intersection(cb_ret.index)
    etf_ret = etf_ret.loc[common_idx].sort_index()
    cb_ret = cb_ret.loc[common_idx].sort_index()

    records = []
    for i in range(ROLLING_WINDOW_MONTHS, len(common_idx) + 1):
        window_idx = common_idx.sort_values()[i - ROLLING_WINDOW_MONTHS: i]
        etf_w = etf_ret.loc[window_idx]
        cb_w = cb_ret.loc[window_idx]
        row = {"window_end": window_idx[-1], "纯ETF夏普": sharpe(etf_w)}
        for w in weights:
            combo = (1 - w) * etf_w + w * cb_w
            row[f"{w:.0%}双低夏普"] = sharpe(combo)
        records.append(row)

    df = pd.DataFrame(records).set_index("window_end")
    if df.empty:
        print(f"\n  [{label}] 窗口数不足{ROLLING_WINDOW_MONTHS}月，无法计算滚动窗口")
        return df

    print(f"\n  [{label}] 滚动{ROLLING_WINDOW_MONTHS}月窗口稳健性检验（n={len(df)}个窗口）：")
    baseline_col = "纯ETF夏普"
    for w in weights:
        col = f"{w:.0%}双低夏普"
        worse_ratio = (df[col] < df[baseline_col]).mean()
        print(f"    权重{w:.0%}：夏普均值={df[col].mean():.3f}(纯ETF均值={df[baseline_col].mean():.3f})  "
              f"跑输纯ETF窗口占比={worse_ratio:.1%}{'（>30%不稳健）' if worse_ratio > 0.3 else ''}")
    return df


def run_weight_sweep(etf_ret: pd.Series, cb_ret: pd.Series, label: str) -> pd.DataFrame:
    common_idx = etf_ret.index.intersection(cb_ret.index)
    etf_ret = etf_ret.loc[common_idx].sort_index()
    cb_ret = cb_ret.loc[common_idx].sort_index()
    print(f"\n对齐后共同月份数：{len(common_idx)}，区间：{common_idx.min().date()} ~ {common_idx.max().date()}")

    corr = etf_ret.corr(cb_ret)
    print(f"月度收益相关系数：{corr:.3f}")

    print("\n" + "=" * 90)
    print(f"不同权重组合的表现（ETF轮动 vs 可转债双低分散配置，{label}）")
    print("=" * 90)

    rows = []
    for cb_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combo_ret = (1 - cb_weight) * etf_ret + cb_weight * cb_ret
        rows.append({
            "双低权重": f"{cb_weight:.0%}",
            "年化收益": annual_return(combo_ret),
            "年化夏普": sharpe(combo_ret),
            "最大回撤": max_drawdown_from_ret(combo_ret),
            "收益std(月)": combo_ret.std(),
        })

    df = pd.DataFrame(rows).set_index("双低权重")
    df_fmt = df.copy()
    df_fmt["年化收益"] = df_fmt["年化收益"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["年化夏普"] = df_fmt["年化夏普"].map(lambda x: f"{x:.3f}")
    df_fmt["最大回撤"] = df_fmt["最大回撤"].map(lambda x: f"{x*100:.1f}%")
    df_fmt["收益std(月)"] = df_fmt["收益std(月)"].map(lambda x: f"{x*100:.2f}%")
    print(df_fmt.to_string())

    baseline_sharpe = df.loc["0%", "年化夏普"]
    best_row = df.drop("0%").sort_values("年化夏普", ascending=False).iloc[0]
    best_weight = df.drop("0%").sort_values("年化夏普", ascending=False).index[0]
    print(f"\n纯ETF轮动（0%双低）夏普={baseline_sharpe:.3f}")
    print(f"含双低最优权重「{best_weight}」夏普={best_row['年化夏普']:.3f}，Δ={best_row['年化夏普']-baseline_sharpe:+.3f}")
    return df


def main():
    etf_cache = RESULTS_DIR / "etf_flow_monthly_ret.csv"
    if etf_cache.exists():
        print(f"复用已缓存的ETF轮动月度收益序列：{etf_cache}")
        etf_ret = pd.read_csv(etf_cache, index_col=0, parse_dates=True)["ret"]
    else:
        etf_ret = build_etf_flow_monthly_returns()
        etf_ret.rename("ret").to_csv(etf_cache)

    print("\n" + "#" * 90)
    print("# 版本1：双低绝对收益（strategy列，含转债市场beta）")
    print("#" * 90)
    cb_abs = load_cb_monthly_returns(use_excess=False)
    df_abs = run_weight_sweep(etf_ret, cb_abs, "绝对收益")
    df_abs.to_csv(RESULTS_DIR / "diversification_weight_sweep_绝对收益.csv")

    print("\n" + "#" * 90)
    print("# 版本2：双低超额收益（strategy-benchmark，剔除转债市场beta，Phase 4b口径）")
    print("#" * 90)
    cb_excess = load_cb_monthly_returns(use_excess=True)
    df_excess = run_weight_sweep(etf_ret, cb_excess, "超额收益")
    df_excess.to_csv(RESULTS_DIR / "diversification_weight_sweep_超额收益.csv")

    print(f"\n结果已保存：{RESULTS_DIR}/diversification_weight_sweep_{{绝对收益,超额收益}}.csv")

    print("\n" + "#" * 90)
    print("# 滚动窗口稳健性检验（权重扫描峰值是否对样本区间敏感）")
    print("#" * 90)
    sweep_weights = [0.1, 0.3, 0.5, 0.8]
    rolling_abs = rolling_diversification_check(etf_ret, cb_abs, sweep_weights, "绝对收益")
    if not rolling_abs.empty:
        rolling_abs.to_csv(RESULTS_DIR / f"rolling_{ROLLING_WINDOW_MONTHS}m_绝对收益.csv")
    rolling_excess = rolling_diversification_check(etf_ret, cb_excess, sweep_weights, "超额收益")
    if not rolling_excess.empty:
        rolling_excess.to_csv(RESULTS_DIR / f"rolling_{ROLLING_WINDOW_MONTHS}m_超额收益.csv")


if __name__ == "__main__":
    main()
