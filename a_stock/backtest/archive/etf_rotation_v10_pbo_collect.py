"""
ETF轮动策略 PBO（Probability of Backtest Overfitting）数据收集
第十轮调研 Task 3：用 runpy 逐个执行历史回测脚本（不改动原脚本），
从其全局命名空间中提取已经算好的净值序列（navs_full/navs 字典或单个 nav 变量），
对齐日期后构建 (n_observations, n_strategies) 日收益矩阵，
供 overfit_detector.py 的 probability_of_backtest_overfitting 使用。

范围说明（本轮明确排除，避免被误读为"覆盖全部历史方向"）：
  - etf_rotation_v3_analysis.py：排除。方向C/D依赖 tushare PE 数据和
    akshare 北向资金数据，脚本顶层无条件拉取外部数据（无论保留哪个 config
    都会触发），且该脚本中唯一无外部依赖的 LW 维度已由
    etf_rotation_v10_lw_convergence.py 单独覆盖。
  - etf_rotation_v4_newfilters.py：排除。依赖 tushare 社融数据和逐只 ETF
    循环拉取的份额数据（无本地缓存，重跑较慢），且此前调研已判定该方向无效。

覆盖范围：R1/R2/R4/R5/R6/R8/R9（quarterly+voltarget+qdii_ic）+ v10 两个
convergence 脚本的基线/边界variant，合计约55个候选策略。
"""

import pathlib
import runpy
import warnings

import matplotlib
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

BACKTEST_DIR = pathlib.Path(__file__).parent

# (脚本文件名, [(变量类型, 变量名), ...])
# 变量类型：series=单个pd.Series；dict=dict[str, pd.Series]；
#          list_of_dict=list[dict]（每个dict含"nav"键，如v9_voltarget的grid_results）
SOURCES = [
    ("etf_rotation_analysis.py", [("series", "nav_full")]),
    ("etf_rotation_v2_analysis.py", [("dict", "navs")]),
    ("etf_rotation_v3b_fullval.py", [("dict", "navs_full")]),
    ("etf_rotation_52wh.py", [("dict", "navs_full")]),
    ("etf_rotation_p2_validate.py", [("dict", "navs_full")]),
    ("etf_rotation_v5_new_directions.py", [("dict", "navs_full")]),
    ("etf_rotation_v9_quarterly.py", [("dict", "navs_full")]),
    ("etf_rotation_v9_voltarget.py", [
        ("series", "nav_base"),
        ("list_of_dict", "grid_results"),
    ]),
    ("etf_rotation_v9_qdii_ic.py", [
        ("series", "nav_full"),
        ("series", "nav_noq"),
    ]),
    ("etf_rotation_v10_lw_convergence.py", [
        ("series", "nav_base_full"),
        ("series", "nav_full"),  # 网格最后一组variant，仅作补充，非穷尽
    ]),
    ("etf_rotation_v10_voltarget_convergence.py", [
        ("series", "nav_base_full"),
        ("series", "nav_full"),
    ]),
]


def collect_navs(script_name: str, targets: list) -> dict:
    """
    用 runpy 在独立命名空间中执行脚本，从结果中提取净值序列。
    脚本本身的 print/plt.savefig 等副作用会照常发生（savefig 目标目录已在
    .gitignore 中排除），但不会污染当前进程的全局状态。
    """
    script_path = BACKTEST_DIR / script_name
    print(f"\n{'=' * 60}\n运行 {script_name} ...")
    ns = runpy.run_path(str(script_path))

    out = {}
    prefix = script_name.replace(".py", "")
    for kind, var_name in targets:
        if var_name not in ns:
            print(f"  [警告] {script_name} 中未找到变量 {var_name}，跳过")
            continue
        val = ns[var_name]
        if kind == "series":
            out[f"{prefix}::{var_name}"] = val
        elif kind == "dict":
            for label, nav in val.items():
                out[f"{prefix}::{label}"] = nav
        elif kind == "list_of_dict":
            for i, row in enumerate(val):
                nav = row.get("nav")
                if nav is None:
                    continue
                tag = f"tv{row.get('target_vol')}_lb{row.get('vol_lookback')}"
                out[f"{prefix}::{tag}"] = nav
    print(f"  提取到 {len(out)} 条净值序列")
    return out


def main():
    all_navs = {}
    for script_name, targets in SOURCES:
        try:
            navs = collect_navs(script_name, targets)
            all_navs.update(navs)
        except Exception as e:
            print(f"  [错误] {script_name} 执行失败：{e}，跳过该脚本")

    print(f"\n{'=' * 60}\n合计收集到 {len(all_navs)} 条策略净值序列")

    # ── 对齐日期，构建收益矩阵 ────────────────────────────
    # 用交集而非并集：PBO(CPCV)要求矩阵无缺失值，取所有策略共同覆盖的
    # 日期区间最稳妥；不同脚本的 close 起点均为 2016-01-01，但终点可能
    # 因数据更新时间不同而略有差异。
    common_index = None
    for nav in all_navs.values():
        idx = nav.dropna().index
        common_index = idx if common_index is None else common_index.intersection(idx)

    print(f"共同日期区间：{common_index.min().date()} ~ {common_index.max().date()}，"
          f"{len(common_index)} 个交易日")

    rets = {}
    for label, nav in all_navs.items():
        aligned = nav.reindex(common_index).ffill()
        rets[label] = aligned.pct_change()

    ret_df = pd.DataFrame(rets).iloc[1:]  # 首行 pct_change 为 NaN
    ret_df = ret_df.dropna(axis=1)  # 丢弃仍有缺失的列（数据起点晚于common_index的极端情况）
    dropped = len(all_navs) - ret_df.shape[1]
    if dropped:
        print(f"  [注意] {dropped} 条序列因对齐后仍含缺失值被丢弃")

    print(f"最终收益矩阵：{ret_df.shape[0]} 个观测 × {ret_df.shape[1]} 个策略")

    out_dir = BACKTEST_DIR.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v10_pbo_returns_matrix.csv"
    ret_df.to_csv(out_path)
    print(f"收益矩阵已保存：{out_path}")


if __name__ == "__main__":
    main()
