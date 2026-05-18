"""
相关性矩阵热力图
计算 11 条策略月度 P&L 的两两相关系数，输出交互热力图
"""
import math
import plotly.graph_objects as go
from analysis_utils import read_monthly_pnl


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


def pearson(a: list[float], b: list[float]) -> float:
    """计算两个等长序列的 Pearson 相关系数。"""
    n = len(a)
    if n < 2:
        return float("nan")
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b))
    if std_a < 1e-9 or std_b < 1e-9:
        return float("nan")
    return cov / (std_a * std_b)


def build_corr_matrix(files: dict[str, str], base: str
                      ) -> tuple[list[str], list[list[float]]]:
    """
    读取所有策略月度 P&L，对齐公共月份，计算相关性矩阵。
    返回 (labels, matrix)
    """
    monthly_data: dict[str, dict[tuple[int, int], float]] = {}
    for key, fname in files.items():
        monthly_data[key] = read_monthly_pnl(base + fname)

    labels = list(files.keys())

    # 找公共月份（取所有策略都有数据的月份）
    all_months = set(monthly_data[labels[0]].keys())
    for key in labels[1:]:
        all_months &= set(monthly_data[key].keys())
    common_months = sorted(all_months)

    if len(common_months) < 6:
        # 公共月份太少，改用并集（缺失月补 0）
        all_months_union = set()
        for key in labels:
            all_months_union |= set(monthly_data[key].keys())
        common_months = sorted(all_months_union)

    # 构建对齐序列
    aligned: dict[str, list[float]] = {}
    for key in labels:
        aligned[key] = [monthly_data[key].get(m, 0.0) for m in common_months]

    # 计算相关性矩阵
    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = pearson(aligned[labels[i]], aligned[labels[j]])

    return labels, matrix, len(common_months)


def plot_correlation(labels: list[str], matrix: list[list[float]],
                     n_months: int, out_path: str):
    """生成相关性热力图。"""
    # 格式化文本
    text = [[f"{matrix[i][j]:.2f}" for j in range(len(labels))]
            for i in range(len(labels))]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[
            [0.0,  "#d73027"],
            [0.25, "#fc8d59"],
            [0.5,  "#ffffbf"],
            [0.75, "#91bfdb"],
            [1.0,  "#4575b4"],
        ],
        zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="相关系数", tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="%{y} × %{x}: %{text}<extra></extra>",
    ))

    # 标注高相关区域（|r| > 0.7）
    annotations = []
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            r = matrix[i][j]
            if i != j and abs(r) > 0.7:
                annotations.append(dict(
                    x=col_label, y=row_label,
                    text=f"<b>{r:.2f}</b>",
                    showarrow=False,
                    font=dict(color="white", size=12),
                ))

    fig.update_layout(
        title=dict(
            text=f"策略月度收益相关性矩阵（基于 {n_months} 个公共月份）",
            font=dict(size=15),
        ),
        height=600,
        width=700,
        template="plotly_white",
        annotations=annotations,
        xaxis=dict(tickangle=45),
    )
    fig.write_html(out_path)
    print(f"  相关性矩阵已保存: {out_path}")

    # 打印高相关对（|r| > 0.6）
    print("\n  高相关策略对（|r| > 0.6）：")
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = matrix[i][j]
            if abs(r) > 0.6:
                pairs.append((abs(r), labels[i], labels[j], r))
    for _, a, b, r in sorted(pairs, reverse=True):
        print(f"    {a} × {b}: {r:.3f}")


def run(files: dict[str, str], base: str, out_dir: str):
    print("计算相关性矩阵...")
    labels, matrix, n_months = build_corr_matrix(files, base)
    plot_correlation(labels, matrix, n_months,
                     f"{out_dir}/相关性矩阵.html")


if __name__ == "__main__":
    import os
    base = os.path.expanduser("~/Downloads/")
    out  = os.path.expanduser("~/Downloads/")
    run(DEFAULT_FILES, base, out)
