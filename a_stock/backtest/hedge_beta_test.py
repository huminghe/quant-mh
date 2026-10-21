"""
股指期货对冲ETF轮动组合beta中性验证

背景：历史调研（research_etf_rotation.md十一轮）排除"股指期货基差作为
市场状态信号"时，曾顺带判断"期货对冲/beta中性"方向"依赖尚未有的选股
alpha"，但这个判断本身从未用数据验证，只是逻辑推断。本次用户要求实测：
用IF.CFX/IC.CFX对431池+flow组合做滚动beta动态对冲，看对冲后剩余收益
是否接近零（若接近零，说明组合收益主要来自beta暴露而非alpha，对冲没有
意义；若对冲后仍有显著正收益，说明存在真实alpha，对冲值得做）。

方法：
1. 现货腿：复用 etf_rotation_45pool_vs_431flow_202607_09.py 的
   build_431_flow_scores + run_backtest，取全历史（2016-至今）净值序列
   （而非仅3个月展示窗口），换算为日收益率序列
2. 对冲腿：IF.CFX（沪深300）、IC.CFX（中证500）主力连续合约日收益率
   （fetch_index_futures.py 已落盘）
3. 滚动beta：60日滚动OLS回归，组合收益 ~ 期货收益，估计动态beta
4. 动态对冲净收益 = 组合收益 - beta * 期货收益（做空期货对冲多头beta
   暴露），按trading-standards.md做样本内外（80/20）拆分对比

已知局限：flow信号依赖的 fund_share 接口单次调用2000条截断、未做offset
分页，实测最早只能拉到2018-07-30，意味着2016-2018年全历史窗口里flow部分
实质退化为无效（boost=NaN，连续打折公式下相当于中性值1.0），这段时期的
现货腿净值序列本质是纯动量信号，不是完整的"431+flow"配置。不影响beta
验证的核心结论（beta本身是价格序列的性质，与flow信号是否生效无关），
但样本早期的"431+flow"命名在严格意义上不准确，特此说明。
"""

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "data"))
from fetch_data import load_close_matrix, init_pro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent / "archive"))
from etf_rotation_v17_new_signal_ic import fetch_fund_share_all  # noqa: E402
from etf_rotation_v18_signal_ablation import (  # noqa: E402
    calc_risk_adj_momentum, run_backtest, calc_stats, get_rebalance_dates,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
CANDIDATES_FILE = DATA_DIR / "etf_all_candidates.parquet"
FUTURES_FILE = DATA_DIR / "index_futures_daily.parquet"
BENCHMARK_CODE = "510300.SH"

FULL_START = "2016-01-01"
ROLL_WINDOW = 60  # 滚动beta估计窗口（交易日）
IS_FRACTION = 0.8  # 样本内/外拆分比例


def build_431_flow_nav():
    codes = pd.read_parquet(CANDIDATES_FILE)["ts_code"].tolist()
    close_full = load_close_matrix(codes + [BENCHMARK_CODE])
    close_full = close_full[close_full.index >= FULL_START]
    valid_codes = [c for c in codes if c in close_full.columns and close_full[c].notna().sum() >= 45]
    close = close_full[valid_codes]
    print(f"431池有效标的：{len(valid_codes)} 只，价格区间 {close.index[0].date()} ~ {close.index[-1].date()}")

    scores = calc_risk_adj_momentum(close)[valid_codes]
    rebal_dates = get_rebalance_dates(close.index)

    print("拉取431池ETF份额数据（flow信号）...")
    pro = init_pro()
    share_matrix = fetch_fund_share_all(pro, valid_codes, start_date=FULL_START.replace("-", ""))
    monthly_share = share_matrix.resample("ME").last() if not share_matrix.empty else pd.DataFrame()
    flow_1m = monthly_share.pct_change() if not monthly_share.empty else pd.DataFrame()
    if not flow_1m.empty:
        print(f"  flow信号覆盖区间：{flow_1m.index.min().date()} ~ {flow_1m.index.max().date()}"
              f"（早于此区间的调仓无flow boost，退化为纯动量）")

    boost_rows = {}
    for d in rebal_dates:
        if flow_1m.empty:
            continue
        idx = flow_1m.index[flow_1m.index <= d]
        if len(idx) == 0:
            continue
        flow_d = flow_1m.loc[idx[-1]].dropna()
        if len(flow_d) < 5:
            continue
        r = flow_d.rank(pct=True)
        boost_rows[d] = 1 - r
    boost_signal = pd.DataFrame(boost_rows).T
    boost_signal.index = pd.to_datetime(boost_signal.index)

    nav = run_backtest(close, scores, rebal_dates, boost_signal=boost_signal, boost_mode="continuous")
    return nav


def load_futures_returns():
    df = pd.read_parquet(FUTURES_FILE)
    out = {}
    for code in df["ts_code"].unique():
        sub = df[df["ts_code"] == code].set_index("trade_date").sort_index()
        out[code] = sub["close"].pct_change()
    return pd.DataFrame(out)


def rolling_beta(port_ret: pd.Series, fut_ret: pd.Series, window: int) -> pd.Series:
    """滚动OLS beta：port_ret ~ beta * fut_ret（单变量回归，截距忽略，只取斜率）"""
    cov = port_ret.rolling(window).cov(fut_ret)
    var = fut_ret.rolling(window).var()
    return cov / var


def split_is_oos(series: pd.Series, frac: float):
    n = len(series)
    cut = int(n * frac)
    return series.iloc[:cut], series.iloc[cut:]


def perf_report(rets: pd.Series, label: str):
    rets = rets.dropna()
    if len(rets) < 30:
        print(f"  {label}：样本不足（{len(rets)}天），跳过")
        return
    nav = (1 + rets).cumprod()
    years = len(rets) / 252
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    rolling_max = nav.cummax()
    max_dd = ((nav - rolling_max) / rolling_max).min()
    print(f"  {label}：年化收益 {cagr:+.2%}，年化夏普 {sharpe:+.3f}，最大回撤 {max_dd:.2%}，天数 {len(rets)}")
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "n": len(rets)}


def main():
    print("=" * 70)
    print("Step 1: 构建431池+flow组合全历史净值序列（现货腿）")
    nav = build_431_flow_nav()
    port_ret = nav.pct_change().dropna()
    print(f"组合净值序列：{port_ret.index[0].date()} ~ {port_ret.index[-1].date()}，{len(port_ret)} 个交易日收益")

    print("\n" + "=" * 70)
    print("Step 2: 加载股指期货收益率序列")
    fut_ret = load_futures_returns()
    print(f"期货品种：{list(fut_ret.columns)}")

    print("\n" + "=" * 70)
    print(f"Step 3: 滚动{ROLL_WINDOW}日beta估计 + 动态对冲")

    results = {}
    for fut_code in ["IF.CFX", "IC.CFX"]:
        common_idx = port_ret.index.intersection(fut_ret.index)
        p = port_ret.loc[common_idx]
        f = fut_ret[fut_code].loc[common_idx]

        beta = rolling_beta(p, f, ROLL_WINDOW)
        hedged_ret = p - beta.shift(1) * f  # beta用前一日估计值，避免用到当天信息

        print(f"\n--- {fut_code} 对冲 ---")
        beta_valid = beta.dropna()
        print(f"  滚动beta：均值 {beta_valid.mean():+.3f}，中位数 {beta_valid.median():+.3f}，"
              f"std {beta_valid.std():.3f}，范围 [{beta_valid.min():+.3f}, {beta_valid.max():+.3f}]")

        is_ret, oos_ret = split_is_oos(hedged_ret.dropna(), IS_FRACTION)
        is_raw, oos_raw = split_is_oos(p.loc[hedged_ret.dropna().index], IS_FRACTION)

        print(f"  样本内（{IS_FRACTION:.0%}）：")
        perf_report(is_raw, "未对冲原始组合")
        is_stats = perf_report(is_ret, "对冲后剩余收益")

        print(f"  样本外（{1 - IS_FRACTION:.0%}）：")
        perf_report(oos_raw, "未对冲原始组合")
        oos_stats = perf_report(oos_ret, "对冲后剩余收益")

        results[fut_code] = {"beta": beta, "hedged_ret": hedged_ret, "is": is_stats, "oos": oos_stats}

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_df = pd.DataFrame({
        "port_ret": port_ret,
        "IF_beta": results["IF.CFX"]["beta"],
        "IF_hedged_ret": results["IF.CFX"]["hedged_ret"],
        "IC_beta": results["IC.CFX"]["beta"],
        "IC_hedged_ret": results["IC.CFX"]["hedged_ret"],
    })
    out_path = out_dir / "hedge_beta_test.csv"
    out_df.to_csv(out_path)
    print(f"\n结果已保存：{out_path}")


if __name__ == "__main__":
    main()
