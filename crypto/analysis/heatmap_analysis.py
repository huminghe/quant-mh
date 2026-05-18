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
    for a in assets:
        keys = [f"{a}_{s}" for s in strats_per.get(a, []) if f"{a}_{s}" in raw]
        if keys:
            combos_built[f"{a}融合"] = merge_avg(*keys)
            combos[f"{a}融合"] = combos_built[f"{a}融合"]

    # 跨标的组合（只在有对应融合时构建）
    fused = [f"{a}融合" for a in assets if f"{a}融合" in combos_built]
    for i in range(len(fused)):
        for j in range(i + 1, len(fused)):
            name = f"{fused[i].replace('融合','')}+{fused[j].replace('融合','')}"
            combos_built[name] = merge_avg(fused[i], fused[j])
            combos[name] = combos_built[name]
    if len(fused) >= 3:
        name3 = "+".join(f.replace("融合", "") for f in fused[:3])
        combos_built[name3] = merge_avg(*fused[:3])
        combos[name3] = combos_built[name3]
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
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=keys,
        vertical_spacing=0.04 / max(n, 1),
    )

    row_heights = []
    for i, key in enumerate(keys, 1):
        monthly = combos.get(key, {})
        years, matrix = monthly_pnl_to_matrix(monthly)
        if not years:
            continue
        hm = make_heatmap(years, matrix, key)
        fig.add_trace(hm, row=i, col=1)
        row_heights.append(len(years))

    # 动态高度：每年约 28px
    total_rows = sum(row_heights) + len(keys) * 3
    height = max(600, total_rows * 28 + len(keys) * 60)

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

    # 图1：11条单策略
    single_keys = [k for k in combos if "_" in k]
    plot_heatmaps(combos, single_keys,
                  f"{out_dir}/热力图_单策略月度收益.html",
                  "各单策略月度收益率热力图")

    # 图2：各标的融合 + 跨标的组合
    combo_keys = ["BTC融合", "ETH融合", "SOL融合", "DOGE融合",
                  "ETH+SOL", "BTC+ETH+SOL", "全4标的"]
    plot_heatmaps(combos, combo_keys,
                  f"{out_dir}/热力图_组合月度收益.html",
                  "各组合月度收益率热力图")


if __name__ == "__main__":
    import os
    base = os.path.expanduser("~/Downloads/")
    out  = os.path.expanduser("~/Downloads/")
    run(DEFAULT_FILES, base, out)
