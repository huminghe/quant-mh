"""
市值加权+主板过滤的截面IC通用工具函数

背景：项目既有20个factor_ic_*.py脚本均对全部指数成分股做等权Spearman
Rank IC，未按主板过滤、未做市值加权。第十九轮deep-research外部调研
（2026-09-18，见docs/research_index_enhancement.md）交叉验证确认Li,
Liu, Liu & Wei《Replicating and Digesting Anomalies in the Chinese
A-Share Market》(Management Science, 2024)的方法论发现：全A股等权分组
会给微小市值/壳股过大权重，系统性高估异象有效性；改用主板+市值加权分组
后，469个类Hou-Xue-Zhang异象变量中83.37%在高低五分组价差上不再显著。

用途：本文件只提供通用函数，不改动任何既有factor_ic_*.py脚本（历史
45类候选IC数值分布无灰色区间，普遍与0.03阈值相差2-3倍以上，不存在
"改分组方法就能翻盘"的候选，回溯重跑无意义，详见lessons.md/memory）。
后续新增候选因子IC检验时，可直接调用本文件的函数代替原有等权版本。

用法示例：
    from factor_ic_weighted import load_total_mv_panel, is_main_board, \
        cross_section_rank_ic_weighted

    mv_panel = load_total_mv_panel()
    main_board_mask = is_main_board(factor.index)
    ic = cross_section_rank_ic_weighted(
        factor[main_board_mask], fwd_ret[main_board_mask],
        weight=mv_panel.loc[month_end, factor.index[main_board_mask]],
    )
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
VALUATION_FILE = DATA_DIR / "valuation_monthly.parquet"

MIN_STOCKS_PER_CROSS = 50


def load_total_mv_panel() -> pd.DataFrame:
    """月度总市值面板（index=trade_date, columns=ts_code），来自fetch_valuation.py产出。"""
    df = pd.read_parquet(VALUATION_FILE)[["trade_date", "ts_code", "total_mv"]].dropna()
    return df.pivot(index="trade_date", columns="ts_code", values="total_mv").sort_index()


def is_main_board(ts_codes: pd.Index) -> pd.Series:
    """
    按ts_code代码段判断是否属于主板（沪市60/深市00开头），
    剔除创业板（30开头）、科创板（688开头）、北交所（8/4开头）。
    返回与输入等长的bool Series，index=ts_codes。
    """
    codes = pd.Series(ts_codes, index=ts_codes).astype(str)
    prefix = codes.str.slice(0, 2)
    is_sh_main = prefix == "60"
    is_sz_main = (prefix == "00") & (~codes.str.slice(0, 3).eq("300"))
    return is_sh_main | is_sz_main


def cross_section_rank_ic_weighted(
    factor: pd.Series,
    fwd_ret: pd.Series,
    weight: pd.Series = None,
) -> float:
    """
    截面Rank IC，可选按市值加权。

    weight=None时退化为普通等权Spearman Rank IC（等价于既有
    factor_ic_*.py脚本的cross_section_rank_ic）。
    weight给定时，先对factor/fwd_ret分别按weight排序算加权秩，
    再算加权Pearson相关系数（等价于市值加权Rank IC）。
    """
    if weight is None:
        aligned = pd.concat([factor, fwd_ret], axis=1).dropna()
        aligned.columns = ["factor", "fwd_ret"]
        if len(aligned) < MIN_STOCKS_PER_CROSS:
            return np.nan
        ic, _ = spearmanr(aligned["factor"], aligned["fwd_ret"])
        return ic

    aligned = pd.concat([factor, fwd_ret, weight], axis=1).dropna()
    aligned.columns = ["factor", "fwd_ret", "weight"]
    if len(aligned) < MIN_STOCKS_PER_CROSS:
        return np.nan
    if (aligned["weight"] <= 0).any():
        aligned = aligned[aligned["weight"] > 0]
        if len(aligned) < MIN_STOCKS_PER_CROSS:
            return np.nan

    def _weighted_rank(s: pd.Series, w: pd.Series) -> pd.Series:
        order = s.sort_values().index
        cum_w = w.loc[order].cumsum()
        total_w = w.sum()
        return (cum_w - 0.5 * w.loc[order]) / total_w

    rank_factor = _weighted_rank(aligned["factor"], aligned["weight"])
    rank_ret = _weighted_rank(aligned["fwd_ret"], aligned["weight"])
    w = aligned["weight"].loc[rank_factor.index]

    def _weighted_pearson(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
        wx_mean = np.average(x, weights=w)
        wy_mean = np.average(y, weights=w)
        cov = np.average((x - wx_mean) * (y - wy_mean), weights=w)
        var_x = np.average((x - wx_mean) ** 2, weights=w)
        var_y = np.average((y - wy_mean) ** 2, weights=w)
        if var_x < 1e-12 or var_y < 1e-12:
            return np.nan
        return cov / np.sqrt(var_x * var_y)

    return _weighted_pearson(rank_factor, rank_ret, w)
