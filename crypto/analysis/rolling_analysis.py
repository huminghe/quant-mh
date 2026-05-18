"""
滚动窗口分析
对每条策略/组合计算滚动 12 个月的：夏普比率、最大回撤、月胜率
输出交互 HTML 图表
"""
import math
import datetime
from analysis_utils import (read_equity_curve, read_monthly_pnl,
                             interp_series, avg_series)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── 配置 ─────────────────────────────────────────────────────────────────────
WINDOW_MONTHS = 12  # 滚动窗口（月）

FILES: dict[str, str] = {}  # 由 run_analysis.py 注入，或直接运行时用下面的默认值

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

ASSET_COLORS = {
    "BTC": "#F7931A", "ETH": "#627EEA",
    "SOL": "#9945FF", "DOGE": "#C2A633",
}
STRAT_DASH = {"ema": "dot", "v2": "dash", "v3": "solid"}


# ─── 滚动指标计算 ──────────────────────────────────────────────────────────────

def rolling_metrics(monthly_pnl: dict[tuple[int, int], float],
                    window: int = 12) -> list[tuple[datetime.date, float, float, float]]:
    """
    计算滚动窗口指标。
    返回 [(end_date, sharpe, max_dd_pct, win_rate), ...]
    monthly_pnl: {(year, month): pnl_usdt}，初始资本 50000
    """
    INITIAL = 50_000.0
    if not monthly_pnl:
        return []

    # 排序月份
    months = sorted(monthly_pnl.keys())
    results = []

    for i in range(window - 1, len(months)):
        window_months = months[i - window + 1: i + 1]
        pnls = [monthly_pnl.get(m, 0.0) for m in window_months]

        # 月度收益率（相对初始资本，简化处理）
        returns = [p / INITIAL for p in pnls]

        # 夏普（年化，假设无风险利率 0）
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
        sharpe = (mean_r / std_r * math.sqrt(12)) if std_r > 1e-9 else 0.0

        # 最大回撤（基于累计 P&L）
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = (peak - cum) / INITIAL * 100
            if dd > max_dd:
                max_dd = dd

        # 月胜率
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100

        end_ym = window_months[-1]
        end_date = datetime.date(end_ym[0], end_ym[1], 1)
        results.append((end_date, sharpe, max_dd, win_rate))

    return results


def build_rolling_data(files: dict[str, str], base: str,
                       window: int = 12) -> dict[str, list]:
    """读取所有策略的滚动指标。"""
    data = {}
    for key, fname in files.items():
        monthly = read_monthly_pnl(base + fname)
        metrics = rolling_metrics(monthly, window)
        if metrics:
            data[key] = {
                "dates":     [m[0] for m in metrics],
                "sharpe":    [m[1] for m in metrics],
                "max_dd":    [m[2] for m in metrics],
                "win_rate":  [m[3] for m in metrics],
            }
    return data


# ─── 绘图 ─────────────────────────────────────────────────────────────────────

def plot_rolling(data: dict, out_path: str, window: int = 12):
    """生成滚动指标三联图（夏普 / 最大回撤 / 月胜率）。"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.31, 0.31],
        vertical_spacing=0.04,
        subplot_titles=[
            f"滚动 {window} 个月夏普比率（年化）",
            f"滚动 {window} 个月最大回撤 %",
            f"滚动 {window} 个月月胜率 %",
        ],
    )

    for key, d in data.items():
        asset = key.split("_")[0]
        strat = key.split("_")[1]
        color = ASSET_COLORS.get(asset, "#888888")
        dash  = STRAT_DASH.get(strat, "solid")
        name  = f"{asset} {strat.upper()}"

        common = dict(x=d["dates"], name=name,
                      line=dict(color=color, dash=dash, width=1.8),
                      legendgroup=asset,
                      legendgrouptitle_text=asset if strat == list(
                          {k.split("_")[1] for k in data if k.startswith(asset)}
                      )[0] else None)

        fig.add_trace(go.Scatter(**common, y=d["sharpe"],  showlegend=True),  row=1, col=1)
        fig.add_trace(go.Scatter(**common, y=d["max_dd"],  showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(**common, y=d["win_rate"],showlegend=False), row=3, col=1)

    # 参考线
    fig.add_hline(y=0,  line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

    fig.update_layout(
        title=dict(text=f"滚动 {window} 个月策略稳定性分析", font=dict(size=15)),
        height=800,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(groupclick="toggleitem", tracegroupgap=6),
    )
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=3, col=1)
    fig.write_html(out_path)
    print(f"  滚动分析图已保存: {out_path}")


# ─── 直接运行入口 ──────────────────────────────────────────────────────────────

def run(files: dict[str, str], base: str, out_dir: str, window: int = 12):
    print("计算滚动指标...")
    data = build_rolling_data(files, base, window)
    plot_rolling(data, f"{out_dir}/滚动窗口分析.html", window)


if __name__ == "__main__":
    import os, sys
    base = os.path.expanduser("~/Downloads/")
    out  = os.path.expanduser("~/Downloads/")
    run(DEFAULT_FILES, base, out)
