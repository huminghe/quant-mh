"""
加密货币趋势策略分析脚本
用法：python strategy_analysis.py <csv文件路径>
输入：TradingView 策略测试器导出的 CSV 文件
输出：整体统计、年度/季度盈亏比、滚动盈亏比、大亏损分析
"""

import sys
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # 只取出场行
    df = df[df["类型"].str.contains("出场")].copy()
    df["日期和时间"] = pd.to_datetime(df["日期和时间"])
    df["年份"] = df["日期和时间"].dt.year
    df["季度"] = df["日期和时间"].dt.to_period("Q")
    df["盈亏"] = df["净损益 %"].apply(lambda x: "盈利" if x > 0 else "亏损")
    return df


def calc_stats(df: pd.DataFrame) -> dict:
    profits = df[df["净损益 %"] > 0]["净损益 %"]
    losses = df[df["净损益 %"] < 0]["净损益 %"].abs()
    avg_profit = profits.mean() if len(profits) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    rr = avg_profit / avg_loss if avg_loss > 0 else 0
    win_rate = len(profits) / len(df) if len(df) > 0 else 0
    expectancy = win_rate * avg_profit - (1 - win_rate) * avg_loss
    return {
        "笔数": len(df),
        "胜率": round(win_rate * 100, 2),
        "平均盈利%": round(avg_profit, 2),
        "平均亏损%": round(avg_loss, 2),
        "盈亏比": round(rr, 3),
        "期望值/笔%": round(expectancy, 3),
        "总盈亏%": round(df["净损益 %"].sum(), 2),
        "总盈亏USDT": round(df["净损益 USDT"].sum(), 0),
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def overall_stats(df: pd.DataFrame):
    print_section("整体统计")
    s = calc_stats(df)
    print(f"总交易笔数：{s['笔数']}")
    print(f"胜率：{s['胜率']}%")
    print(f"平均盈利：{s['平均盈利%']}%  |  平均亏损：{s['平均亏损%']}%")
    print(f"盈亏比：{s['盈亏比']}")
    print(f"期望值/笔：{s['期望值/笔%']}%")
    print(f"总盈亏：{s['总盈亏USDT']} USDT（{s['总盈亏%']}%）")


def yearly_stats(df: pd.DataFrame):
    print_section("年度统计")
    print(f"{'年份':<8}{'笔数':<8}{'胜率%':<10}{'平均盈利%':<12}{'平均亏损%':<12}{'盈亏比':<10}{'总盈亏USDT'}")
    print("-" * 70)
    for year, group in df.groupby("年份"):
        s = calc_stats(group)
        print(f"{year:<8}{s['笔数']:<8}{s['胜率']:<10}{s['平均盈利%']:<12}{s['平均亏损%']:<12}{s['盈亏比']:<10}{s['总盈亏USDT']}")


def quarterly_stats(df: pd.DataFrame):
    print_section("季度统计")
    print(f"{'季度':<12}{'笔数':<8}{'胜率%':<10}{'平均盈利%':<12}{'平均亏损%':<12}{'盈亏比':<10}{'总盈亏%'}")
    print("-" * 72)
    for quarter, group in df.groupby("季度"):
        s = calc_stats(group)
        print(f"{str(quarter):<12}{s['笔数']:<8}{s['胜率']:<10}{s['平均盈利%']:<12}{s['平均亏损%']:<12}{s['盈亏比']:<10}{s['总盈亏%']}")


def rolling_rr(df: pd.DataFrame, window: int = 30):
    print_section(f"滚动{window}笔盈亏比")
    df = df.reset_index(drop=True)
    rr_series = []
    for i in range(window - 1, len(df)):
        window_df = df.iloc[i - window + 1: i + 1]
        s = calc_stats(window_df)
        rr_series.append({
            "结束日期": df.iloc[i]["日期和时间"].strftime("%Y-%m-%d"),
            "盈亏比": s["盈亏比"],
            "胜率%": s["胜率"],
        })
    rr_df = pd.DataFrame(rr_series)
    hist_mean = rr_df["盈亏比"].mean()
    hist_min = rr_df["盈亏比"].min()
    hist_p10 = rr_df["盈亏比"].quantile(0.1)
    current = rr_df.iloc[-1]

    print(f"历史均值：{round(hist_mean, 3)}")
    print(f"历史10%分位：{round(hist_p10, 3)}")
    print(f"历史最低：{round(hist_min, 3)}（{rr_df.loc[rr_df['盈亏比'].idxmin(), '结束日期']}）")
    print(f"当前（最近{window}笔）：{current['盈亏比']}（截至{current['结束日期']}）")
    print(f"预警线（均值-30%）：{round(hist_mean * 0.7, 3)}")

    # 最近10个窗口
    print(f"\n最近10个窗口趋势：")
    for _, row in rr_df.tail(10).iterrows():
        bar = "█" * int(row["盈亏比"] * 5)
        print(f"  {row['结束日期']}  {row['盈亏比']:.3f}  {bar}")


def large_loss_analysis(df: pd.DataFrame, threshold: float = -5.0):
    print_section(f"大亏损分析（净损益% < {threshold}%）")
    large_losses = df[df["净损益 %"] < threshold].copy()
    large_losses = large_losses.sort_values("净损益 %")
    print(f"共 {len(large_losses)} 笔，占总交易 {round(len(large_losses)/len(df)*100, 1)}%\n")
    print(f"{'日期':<22}{'方向':<10}{'净损益%':<12}{'净损益USDT'}")
    print("-" * 55)
    for _, row in large_losses.iterrows():
        direction = "多头" if "多头" in row["类型"] else "空头"
        print(f"{str(row['日期和时间']):<22}{direction:<10}{row['净损益 %']:<12}{row['净损益 USDT']}")

    print(f"\n按年分布：")
    for year, group in large_losses.groupby("年份"):
        print(f"  {year}：{len(group)} 笔")


def trend_analysis(df: pd.DataFrame):
    print_section("盈亏比长期趋势（线性回归）")
    yearly = []
    for year, group in df.groupby("年份"):
        if year == df["年份"].max() and len(group) < 50:
            continue  # 跳过不完整年份
        s = calc_stats(group)
        yearly.append({"年份": year, "盈亏比": s["盈亏比"], "胜率": s["胜率"]})
    yearly_df = pd.DataFrame(yearly)

    x = yearly_df["年份"].values - yearly_df["年份"].min()
    y_rr = yearly_df["盈亏比"].values
    y_wr = yearly_df["胜率"].values

    slope_rr = np.polyfit(x, y_rr, 1)[0]
    slope_wr = np.polyfit(x, y_wr, 1)[0]

    print(f"盈亏比年度斜率：{round(slope_rr, 3)}/年（{'上升' if slope_rr > 0 else '下降'}）")
    print(f"胜率年度斜率：{round(slope_wr, 3)}%/年（{'上升' if slope_wr > 0 else '下降'}）")
    print(f"\n结论：{'无系统性衰退迹象' if slope_rr > -0.1 else '盈亏比有下降趋势，需关注'}")


def main():
    if len(sys.argv) < 2:
        print("用法：python strategy_analysis.py <csv文件路径>")
        print("示例：python strategy_analysis.py ~/Downloads/strategy_sol.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n加载文件：{filepath}")

    df = load_data(filepath)
    print(f"数据范围：{df['日期和时间'].min().date()} ~ {df['日期和时间'].max().date()}")

    overall_stats(df)
    yearly_stats(df)
    quarterly_stats(df)
    rolling_rr(df, window=30)
    large_loss_analysis(df, threshold=-5.0)
    trend_analysis(df)

    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
