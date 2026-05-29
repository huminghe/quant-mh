"""
HMM 市场制度监控

从 OKX 拉取 BTC 日线数据，用 2 状态 HMM 识别 risk_on / risk_off 制度，
输出当前制度状态和历史制度序列。

⚠️  已知局限性（2026-05-27 验证）：
OKX 单次最多返回约 300 天数据，且当前处于牛市周期，导致 HMM 训练数据
几乎全是上涨行情，risk_on 占比约 93%，risk_off 识别几乎无意义。
HMM 本质上是低波动/高波动分类器，不是真正的制度识别。

对双向趋势策略而言，BTC 价格方向信号本身也无效——大跌时做空同样盈利，
策略真正的敌人是横盘（低ADX+低ATR），而非价格下跌。

建议：此脚本作为参考输出保留，不应作为暂停决策的依据。
暂停决策仍以现有 Stoch+BBWP+12%跌幅 机制为准。

用法：
  python regime_monitor.py              # 输出当前制度状态
  python regime_monitor.py --plot       # 同时生成制度图表
  python regime_monitor.py --days 365   # 使用最近 N 天数据（默认 500）
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    print("缺少 ccxt，请运行：pip install ccxt --break-system-packages")
    sys.exit(1)

try:
    from hmmlearn import hmm
except ImportError:
    print("缺少 hmmlearn，请运行：pip install hmmlearn --break-system-packages")
    sys.exit(1)


# ─── 配置 ─────────────────────────────────────────────────────────────────────

SYMBOL = "BTC/USDT:USDT"   # OKX 永续合约
TIMEFRAME = "1d"
DEFAULT_DAYS = 500          # 训练数据天数
N_STATES = 2                # risk_on / risk_off
RANDOM_SEED = 42


# ─── 数据获取 ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(days: int) -> pd.DataFrame:
    """从 OKX 拉取 BTC 日线数据。"""
    exchange = ccxt.okx({"enableRateLimit": True})
    limit = min(days, 1000)  # OKX 单次最多 1000 根

    print(f"从 OKX 拉取 BTC 日线数据（最近 {limit} 天）...")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    df = df.set_index("date").drop(columns=["timestamp"])
    df = df.sort_index()

    print(f"获取到 {len(df)} 根日线，{df.index[0]} ~ {df.index[-1]}")
    return df


# ─── HMM 制度识别 ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    构建 HMM 特征：
    - 日收益率（捕捉方向）
    - 20 日滚动波动率（捕捉波动环境）
    两个特征组合能区分"趋势上涨"、"趋势下跌"、"低波动横盘"、"高波动崩盘"。
    """
    returns = df["close"].pct_change().fillna(0)
    volatility = returns.rolling(20).std().fillna(returns.std())
    features = np.column_stack([returns.values, volatility.values])
    return features


def fit_hmm(features: np.ndarray) -> tuple[hmm.GaussianHMM, np.ndarray]:
    """拟合 2 状态 HMM，返回模型和状态序列。"""
    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
    )
    model.fit(features)
    states = model.predict(features)
    return model, states


def label_states(model: hmm.GaussianHMM, states: np.ndarray) -> dict[int, str]:
    """
    按各状态的平均收益率排序：
    高均值收益 → risk_on，低均值收益 → risk_off。
    """
    mean_returns = model.means_[:, 0]  # 第 0 列是收益率
    risk_on_state = int(np.argmax(mean_returns))
    risk_off_state = int(np.argmin(mean_returns))
    return {risk_on_state: "risk_on", risk_off_state: "risk_off"}


def get_regime_series(df: pd.DataFrame, states: np.ndarray,
                      state_labels: dict[int, str]) -> pd.Series:
    """返回带日期索引的制度序列。"""
    regime = pd.Series(
        [state_labels[s] for s in states],
        index=df.index,
        name="regime",
    )
    return regime


# ─── 统计分析 ─────────────────────────────────────────────────────────────────

def compute_regime_stats(df: pd.DataFrame, regime: pd.Series) -> dict:
    """计算各制度下的收益统计。"""
    returns = df["close"].pct_change()
    stats = {}
    for label in ["risk_on", "risk_off"]:
        mask = regime == label
        r = returns[mask]
        stats[label] = {
            "days": int(mask.sum()),
            "pct": f"{mask.mean() * 100:.1f}%",
            "mean_daily_return": f"{r.mean() * 100:.2f}%",
            "volatility": f"{r.std() * 100:.2f}%",
            "win_rate": f"{(r > 0).mean() * 100:.1f}%",
        }
    return stats


def get_current_streak(regime: pd.Series) -> tuple[str, int]:
    """返回当前制度和连续天数。"""
    current = regime.iloc[-1]
    streak = 1
    for r in reversed(regime.iloc[:-1].values):
        if r == current:
            streak += 1
        else:
            break
    return current, streak


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_regime(df: pd.DataFrame, regime: pd.Series, out_dir: Path):
    """生成制度图表，保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("缺少 matplotlib，跳过图表生成")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # 上图：BTC 价格 + 制度背景色
    dates = pd.to_datetime(df.index)
    ax1.plot(dates, df["close"], color="#1f77b4", linewidth=1.2, label="BTC 收盘价")
    ax1.set_ylabel("价格 (USDT)")
    ax1.set_title("BTC 市场制度识别（HMM 2状态）")
    ax1.grid(alpha=0.3)

    # 制度背景色
    colors = {"risk_on": "#d4edda", "risk_off": "#f8d7da"}
    prev_regime = None
    start_idx = 0
    for i, (date, r) in enumerate(zip(dates, regime)):
        if r != prev_regime:
            if prev_regime is not None:
                ax1.axvspan(dates[start_idx], date,
                            alpha=0.3, color=colors[prev_regime], linewidth=0)
            start_idx = i
            prev_regime = r
    if prev_regime:
        ax1.axvspan(dates[start_idx], dates[-1],
                    alpha=0.3, color=colors[prev_regime], linewidth=0)

    patches = [
        mpatches.Patch(color="#d4edda", alpha=0.6, label="risk_on"),
        mpatches.Patch(color="#f8d7da", alpha=0.6, label="risk_off"),
    ]
    ax1.legend(handles=patches + [ax1.lines[0]], loc="upper left")

    # 下图：制度状态（0/1）
    regime_numeric = (regime == "risk_on").astype(int)
    ax2.fill_between(dates, regime_numeric, alpha=0.6,
                     color="#2196F3", step="post")
    ax2.set_ylabel("制度状态")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["risk_off", "risk_on"])
    ax2.set_xlabel("日期")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"regime_monitor_{datetime.now().strftime('%Y-%m-%d')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{out_path}")


# ─── 输出报告 ─────────────────────────────────────────────────────────────────

def print_report(regime: pd.Series, stats: dict, current: str, streak: int,
                 model: hmm.GaussianHMM):
    """打印制度监控报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    divider = "=" * 55

    print(f"\n{divider}")
    print(f"  BTC 市场制度监控报告  {now}")
    print(divider)

    # 当前状态
    icon = "✅" if current == "risk_on" else "⚠️"
    print(f"\n{icon} 当前制度：{current.upper()}（已持续 {streak} 天）")

    # 与暂停机制的关联
    print("\n── 与暂停机制的关联 ──")
    if current == "risk_off":
        print("  → 制度信号支持暂停/缩仓")
        print("  → 建议结合 Stoch 信号综合判断")
    else:
        print("  → 制度信号正常，无需暂停")
        if streak < 10:
            print(f"  → 注意：risk_on 仅持续 {streak} 天，制度尚不稳定")

    # 历史统计
    print("\n── 历史制度统计 ──")
    for label, s in stats.items():
        print(f"\n  {label}（{s['days']} 天，占比 {s['pct']}）")
        print(f"    日均收益：{s['mean_daily_return']}")
        print(f"    日波动率：{s['volatility']}")
        print(f"    胜率：    {s['win_rate']}")

    # 最近 30 天制度
    recent = regime.tail(30)
    risk_off_days = (recent == "risk_off").sum()
    print(f"\n── 最近 30 天 ──")
    print(f"  risk_off 天数：{risk_off_days} / 30")
    if risk_off_days >= 15:
        print("  ⚠️  近期 risk_off 占比超过 50%，市场处于防御状态")

    # HMM 转移矩阵
    labels = ["risk_on" if model.means_[i, 0] > model.means_[1-i, 0]
              else "risk_off" for i in range(2)]
    print(f"\n── HMM 状态转移概率 ──")
    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            prob = model.transmat_[i, j]
            if prob > 0.05:
                print(f"  {from_label} → {to_label}：{prob:.1%}")

    print(f"\n{divider}\n")


def save_report(regime: pd.Series, stats: dict, current: str, streak: int,
                out_dir: Path) -> Path:
    """保存 MD 格式报告。"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    status_icon = "risk_on" if current == "risk_on" else "risk_off"
    recent = regime.tail(30)
    risk_off_days_30 = int((recent == "risk_off").sum())

    lines = [
        f"# BTC 市场制度监控 {date_str}",
        "",
        f"> 生成时间：{time_str}  |  数据来源：OKX BTC/USDT 永续合约日线",
        "",
        "---",
        "",
        "## 当前状态",
        "",
        f"**制度：{current.upper()}**（已持续 {streak} 天）",
        "",
        f"近 30 天 risk_off 占比：{risk_off_days_30}/30 天",
        "",
        "## 与暂停机制的关联",
        "",
    ]

    if current == "risk_off":
        lines += [
            "- 制度信号支持暂停/缩仓",
            "- 建议结合 Stoch 信号综合判断",
        ]
    else:
        lines += [
            "- 制度信号正常，无需暂停",
        ]
        if streak < 10:
            lines.append(f"- 注意：risk_on 仅持续 {streak} 天，制度尚不稳定")

    lines += [
        "",
        "## 历史制度统计",
        "",
        "| 制度 | 天数 | 占比 | 日均收益 | 日波动率 | 胜率 |",
        "|------|------|------|---------|---------|------|",
    ]
    for label, s in stats.items():
        lines.append(
            f"| {label} | {s['days']} | {s['pct']} | "
            f"{s['mean_daily_return']} | {s['volatility']} | {s['win_rate']} |"
        )

    lines += [
        "",
        "---",
        "",
        "> 本文件由 `regime_monitor.py` 自动生成。",
        "> HMM 2状态模型，特征：日收益率 + 20日滚动波动率。",
    ]

    out_path = out_dir / f"regime_monitor_{date_str}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存：{out_path}")
    return out_path


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="HMM 市场制度监控")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"使用最近 N 天数据（默认 {DEFAULT_DAYS}）")
    parser.add_argument("--plot", action="store_true",
                        help="生成制度图表（PNG）")
    parser.add_argument("--out", default=str(base_dir),
                        help="输出目录（默认脚本所在目录）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 拉数据
    df = fetch_ohlcv(args.days)

    # 拟合 HMM
    print("拟合 HMM 模型...")
    features = build_features(df)
    model, states = fit_hmm(features)
    state_labels = label_states(model, states)
    regime = get_regime_series(df, states, state_labels)

    # 统计
    stats = compute_regime_stats(df, regime)
    current, streak = get_current_streak(regime)

    # 输出
    print_report(regime, stats, current, streak, model)
    save_report(regime, stats, current, streak, out_dir)

    if args.plot:
        plot_regime(df, regime, out_dir)


if __name__ == "__main__":
    main()
