"""
月度收益热力图
对每条策略/组合生成 年×月 矩阵热力图，直观展示各月收益分布
"""
import datetime
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis_utils import read_monthly_pnl, avg_series


INITIAL = 50_000.0

MONTHS_CN = ["1月", "2月", "3月", "4月", "5月", "6月",
             "7月", "8月", "9月", "10月", "11月", "12月"]

DEFAULT_FILES = {
    "BTC_ema":  "strategy_ema_btc_OKX_BTCUSDT.P_2026-05-18_bccdc.xlsx",
    "BTC_v2":   "v2_strategy_btc_OKX_BTCUSDT.P_2026-05-18_89f2b.xlsx",
    "BTC_v3":   "v3_205m_strategy_btc_OKX_BTCUSDT.P_2026-05-18_7f0bc.xlsx",
    "ETH_ema":  "strategy_ema_eth_OKX_ETHUSDT.P_2026-05-18_6d950.xlsx",
    "ETH_v2":   "v2_strategy_eth_OKX_ETHUSDT.P_2026-05-18_7eba3.xlsx",
    "ETH_v3":   "v3_3h_strategy_eth_OKX_ETHUSDT.P_2026-05-18_f84bb.xlsx",
    "SOL_ema":  "strategy_ema_sol_OKX_SOLUSDT.P_2026-05-18_f737f.xlsx",
    "SOL_v2":   "v2_strategy_sol_OKX_SOLUSDT.P_2026-05-18_2c3a0.xlsx",
    "SOL_v3":   "v3_3h_strategy_sol_OKX_SOLUSDT.P_2026-05-18_a51fb.xlsx",
    "DOGE_v2":  "v2_strategy_doge_OKX_DOGEUSDT.P_2026-05-18_2b9fa.xlsx",
    "DOGE_v3":  "v3_205m_strategy_doge_OKX_DOGEUSDT.P_2026-05-18_283df.xlsx",
}


def monthly_pnl_to_matrix(monthly: dict[tuple[int, int], float]
                           ) -> tuple[list[int], list[list[float | None]]]:
    """
    将 {(year, month): pnl} 转为 年×月 矩阵（收益率%）。
    返回 (years, matrix)，matrix[i][j] = year[i] 第 j+1 月的收益率%
    """
    if not monthly:
        return [], []
    years = sorted(set(y for y, _ in monthly.keys()))
    matrix = []
    for y in years:
        row = []
        for m in range(1, 13):
            pnl = monthly.get((y, m))
            row.append(round(pnl / INITIAL * 100, 2) if pnl is not None else None)
        matrix.append(row)
    return years, matrix


def build_combo_monthly(files: dict[str, str], base: str
                        ) -> dict[str, dict[tuple[int, int], float]]:
    """读取所有策略月度数据，并构建融合组合。"""
    raw: dict[str, dict[tuple[int, int], float]] = {}
    for key, fname in files.items():
        raw[key] = read_monthly_pnl(base + fname)

    # 各标的融合（等权平均 USDT）
    def merge_avg(*keys):
        # 只合并实际存在的 key
        valid_keys = [k for k in keys if k in raw or k in combos_built]
        if not valid_keys:
            return {}
        all_ym = set()
        for k in valid_keys:
            src = raw if k in raw else combos_built
            all_ym |= set(src[k].keys())
        result = {}
        for ym in all_ym:
            vals = [(raw if k in raw else combos_built)[k].get(ym, 0.0)
                    for k in valid_keys]
            result[ym] = sum(vals) / len(vals)
        return result

    # 动态构建各标的融合
    combos_built: dict = {}
    assets = sorted(set(k.split("_")[0] for k in files))
    strats_per: dict[str, list[str]] = {}
    for k in files:
        a, s = k.split("_", 1)
        strats_per.setdefault(a, []).append(s)

    combos = dict(raw)
    single_asset_mode = len(assets) == 1

    if single_asset_mode:
        # 单标的：每个版本独立，直接构建版本间组合
        a = assets[0]
        strats = strats_per.get(a, [])
        units = {s: raw[f"{a}_{s}"] for s in strats if f"{a}_{s}" in raw}
        unit_list = list(units.keys())
        # 两两
        for i in range(len(unit_list)):
            for j in range(i + 1, len(unit_list)):
                u1, u2 = unit_list[i], unit_list[j]
                name = f"{u1}+{u2}"
                combos_built[name] = merge_avg(f"{a}_{u1}", f"{a}_{u2}")
                combos[name] = combos_built[name]
        # 三版本
        for i in range(len(unit_list)):
            for j in range(i + 1, len(unit_list)):
                for k in range(j + 1, len(unit_list)):
                    u1, u2, u3 = unit_list[i], unit_list[j], unit_list[k]
                    name = f"{u1}+{u2}+{u3}"
                    combos_built[name] = merge_avg(f"{a}_{u1}", f"{a}_{u2}", f"{a}_{u3}")
                    combos[name] = combos_built[name]
        # 全版本
        if len(unit_list) >= 4:
            combos_built["全版本等权"] = merge_avg(*[f"{a}_{s}" for s in unit_list])
            combos["全版本等权"] = combos_built["全版本等权"]
    else:
        # 多标的：先按标的融合，再跨标的组合
        for a in assets:
            keys = [f"{a}_{s}" for s in strats_per.get(a, []) if f"{a}_{s}" in raw]
            if keys:
                combos_built[f"{a}融合"] = merge_avg(*keys)
                combos[f"{a}融合"] = combos_built[f"{a}融合"]

        fused = [f"{a}融合" for a in assets if f"{a}融合" in combos_built]
        for i in range(len(fused)):
            for j in range(i + 1, len(fused)):
                name = f"{fused[i].replace('融合','')}+{fused[j].replace('融合','')}"
                combos_built[name] = merge_avg(fused[i], fused[j])
                combos[name] = combos_built[name]
        for i in range(len(fused)):
            for j in range(i + 1, len(fused)):
                for k in range(j + 1, len(fused)):
                    name = "+".join(f.replace("融合", "") for f in [fused[i], fused[j], fused[k]])
                    combos_built[name] = merge_avg(fused[i], fused[j], fused[k])
                    combos[name] = combos_built[name]
        if len(fused) >= 4:
            combos_built["全4标的"] = merge_avg(*fused)
            combos["全4标的"] = combos_built["全4标的"]

    return combos


def make_heatmap(years: list[int], matrix: list[list],
                 title: str) -> go.Heatmap:
    """构建单个热力图 trace。"""
    # 替换 None 为 NaN 字符串用于显示
    z = []
    text = []
    for row in matrix:
        z_row, t_row = [], []
        for v in row:
            if v is None:
                z_row.append(float("nan"))
                t_row.append("")
            else:
                z_row.append(v)
                t_row.append(f"{v:+.1f}%")
        z.append(z_row)
        text.append(t_row)

    return go.Heatmap(
        z=z,
        x=MONTHS_CN,
        y=[str(y) for y in years],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[
            [0.0,  "#d73027"],
            [0.35, "#fc8d59"],
            [0.45, "#fee090"],
            [0.5,  "#ffffbf"],
            [0.55, "#e0f3f8"],
            [0.65, "#91bfdb"],
            [1.0,  "#4575b4"],
        ],
        zmid=0,
        colorbar=dict(title="%", ticksuffix="%", len=0.8),
        name=title,
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
    )


def plot_heatmaps(combos: dict[str, dict[tuple[int, int], float]],
                  keys: list[str], out_path: str, page_title: str):
    """将多个策略/组合的热力图拼成一个 HTML 文件。"""
    n = len(keys)
    # 每个热力图固定高度（px），标题行 40px + 每年 32px
    per_heights = []
    for key in keys:
        monthly = combos.get(key, {})
        years, _ = monthly_pnl_to_matrix(monthly)
        per_heights.append(40 + len(years) * 32)

    total_height = sum(per_heights) + n * 60  # 60px 间距/标题
    height = max(600, total_height)

    # vertical_spacing 用绝对像素换算为比例，最小 0.03
    spacing = max(0.03, 60 / height)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=keys,
        vertical_spacing=spacing,
        row_heights=per_heights,
    )

    for i, key in enumerate(keys, 1):
        monthly = combos.get(key, {})
        years, matrix = monthly_pnl_to_matrix(monthly)
        if not years:
            continue
        hm = make_heatmap(years, matrix, key)
        fig.add_trace(hm, row=i, col=1)

    fig.update_layout(
        title=dict(text=page_title, font=dict(size=15)),
        height=height,
        template="plotly_white",
    )
    fig.write_html(out_path)
    print(f"  热力图已保存: {out_path}")


def run(files: dict[str, str], base: str, out_dir: str):
    print("构建月度数据...")
    combos = build_combo_monthly(files, base)

    # 图1：单策略（只取原始文件对应的 key，不含任何组合）
    single_keys = [k for k in files if k in combos]
    plot_heatmaps(combos, single_keys,
                  f"{out_dir}/热力图_单策略月度收益.html",
                  "各单策略月度收益率热力图")

    # 图2：所有组合（融合 + 两两 + 三标的 + 全组合），按组合层级排序
    fusion_keys  = [k for k in combos if "融合" in k]
    pair_keys    = [k for k in combos if k.count("+") == 1 and "融合" not in k]
    triple_keys  = [k for k in combos if k.count("+") == 2 and "融合" not in k]
    full_keys    = [k for k in combos if k in ("全4标的", "全版本等权")]
    # 单标的模式：把所有版本等权也加进来（key 含 + 且覆盖全部版本）
    n_versions   = len(files)
    if n_versions >= 4:
        full_keys += [k for k in combos if k.count("+") == n_versions - 1
                      and k not in full_keys]
    combo_keys   = fusion_keys + pair_keys + triple_keys + full_keys
    combo_keys   = [k for k in combo_keys if k in combos]  # 过滤不存在的
    if combo_keys:
        plot_heatmaps(combos, combo_keys,
                      f"{out_dir}/热力图_组合月度收益.html",
                      "各组合月度收益率热力图")


if __name__ == "__main__":
    import os
    base = os.path.expanduser("~/Downloads/")
    out  = os.path.expanduser("~/Downloads/")
    run(DEFAULT_FILES, base, out)
