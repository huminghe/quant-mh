"""
20 日短期反转策略 - 沪深 300 成分股
策略逻辑：买入过去 20 日收益率最低的 30% 股票（反转效应）
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 配置参数
# ============================================================================

# Tushare Token（需要替换为你的 token）
TS_TOKEN = 'YOUR_TUSHARE_TOKEN'  # 请在 https://tushare.pro 获取
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 回测参数
START_DATE = '20190101'
END_DATE = '20241231'
LOOKBACK_PERIOD = 20  # 反转因子回看期
QUANTILE_THRESHOLD = 0.3  # 买入收益率最低的 30%
REBALANCE_FREQ = 'M'  # 月度调仓
FEES = 0.00164  # 交易成本：0.164%

# ============================================================================
# 2. 数据获取
# ============================================================================

print("=" * 60)
print("开始获取数据...")
print("=" * 60)

# 2.1 获取沪深 300 成分股列表
def get_hs300_stocks(date=None):
    """获取沪深 300 成分股列表"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    df = pro.index_weight(index_code='399300.SZ', start_date=date, end_date=date)
    if df.empty:
        # 如果指定日期没有数据，获取最近的成分股
        df = pro.index_weight(index_code='399300.SZ')
        df = df.sort_values('trade_date', ascending=False).head(300)

    stocks = df['con_code'].unique().tolist()
    print(f"获取到 {len(stocks)} 只沪深 300 成分股")
    return stocks

# 2.2 获取股票日线数据
def get_stock_data(stocks, start_date, end_date):
    """批量获取股票日线数据"""
    all_data = []

    for i, stock in enumerate(stocks):
        try:
            df = ts.pro_bar(
                ts_code=stock,
                asset='E',
                start_date=start_date,
                end_date=end_date,
                adj='qfq'  # 前复权
            )

            if df is not None and not df.empty:
                df['ts_code'] = stock
                all_data.append(df)

            if (i + 1) % 50 == 0:
                print(f"已获取 {i + 1}/{len(stocks)} 只股票数据")

        except Exception as e:
            print(f"获取 {stock} 数据失败: {e}")
            continue

    if not all_data:
        raise ValueError("未获取到任何股票数据，请检查 Tushare Token 是否正确")

    # 合并数据
    df_all = pd.concat(all_data, ignore_index=True)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    df_all = df_all.sort_values(['ts_code', 'trade_date'])

    print(f"数据获取完成，共 {len(df_all)} 条记录")
    return df_all

# 获取数据
stocks = get_hs300_stocks()
df_raw = get_stock_data(stocks, START_DATE, END_DATE)

# ============================================================================
# 3. 数据预处理
# ============================================================================

print("\n" + "=" * 60)
print("数据预处理...")
print("=" * 60)

# 3.1 转换为宽表格式（日期 x 股票）
close_prices = df_raw.pivot(index='trade_date', columns='ts_code', values='close')
close_prices = close_prices.sort_index()

# 3.2 填充缺失值（使用前值填充，避免未来信息泄露）
close_prices = close_prices.ffill()  # 新版 pandas 语法

# 3.3 过滤数据不足的股票（至少需要 LOOKBACK_PERIOD + 30 天数据）
min_data_points = LOOKBACK_PERIOD + 30
valid_stocks = close_prices.count() >= min_data_points
close_prices = close_prices.loc[:, valid_stocks]

print(f"有效股票数量: {close_prices.shape[1]}")
print(f"回测交易日数量: {close_prices.shape[0]}")
print(f"数据时间范围: {close_prices.index[0]} 至 {close_prices.index[-1]}")

# ============================================================================
# 4. 因子计算
# ============================================================================

print("\n" + "=" * 60)
print("计算反转因子...")
print("=" * 60)

# 4.1 计算 20 日收益率
returns_20d = close_prices.pct_change(LOOKBACK_PERIOD)

# 4.2 计算涨跌停标记（用于过滤）
daily_returns = close_prices.pct_change()
limit_up = daily_returns >= 0.099  # 涨停（接近 10%）
limit_down = daily_returns <= -0.099  # 跌停

print(f"因子计算完成")
print(f"因子数据形状: {returns_20d.shape}")

# ============================================================================
# 5. 生成交易信号
# ============================================================================

print("\n" + "=" * 60)
print("生成交易信号...")
print("=" * 60)

# 5.1 月度调仓日期（ME = Month End，新版 pandas 语法）
rebalance_dates = close_prices.resample('ME').last().index

# 5.2 生成信号矩阵
signals = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)

for date in rebalance_dates:
    if date not in returns_20d.index:
        continue

    # 获取该日期的 20 日收益率
    factor_values = returns_20d.loc[date]

    # 过滤掉 NaN 值
    factor_values = factor_values.dropna()

    if len(factor_values) == 0:
        continue

    # 计算分位数阈值（收益率最低的 30%）
    threshold = factor_values.quantile(QUANTILE_THRESHOLD)

    # 生成买入信号（收益率低于阈值）
    buy_signals = factor_values <= threshold

    # 过滤涨停股（买不进）
    if date in limit_up.index:
        buy_signals = buy_signals & ~limit_up.loc[date]

    # 将信号填充到下个调仓日
    next_rebalance_idx = rebalance_dates.get_loc(date) + 1
    if next_rebalance_idx < len(rebalance_dates):
        end_date = rebalance_dates[next_rebalance_idx]
    else:
        end_date = close_prices.index[-1]

    # 填充信号
    mask = (signals.index >= date) & (signals.index <= end_date)
    signals.loc[mask, buy_signals.index[buy_signals]] = 1

# 5.3 应用 T+1 约束（信号延迟一天执行）
signals = signals.shift(1).fillna(0)

print(f"交易信号生成完成")
print(f"调仓次数: {len(rebalance_dates)}")
print(f"平均持仓股票数: {signals.sum(axis=1).mean():.1f}")

# ============================================================================
# 6. 回测执行
# ============================================================================

print("\n" + "=" * 60)
print("执行回测...")
print("=" * 60)

# 6.1 计算持仓权重（等权重）
weights = signals.div(signals.sum(axis=1), axis=0).fillna(0)

# 6.2 使用 vectorbt 执行回测
portfolio = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=signals > 0,
    exits=signals == 0,
    size=weights,
    size_type='targetpercent',
    fees=FEES,
    freq='D',
    init_cash=1000000  # 初始资金 100 万
)

print("回测执行完成")

# ============================================================================
# 7. 性能分析
# ============================================================================

print("\n" + "=" * 60)
print("性能指标")
print("=" * 60)

# 7.1 基础指标
total_return = portfolio.total_return() * 100
annual_return = portfolio.annualized_return() * 100
sharpe_ratio = portfolio.sharpe_ratio()
max_drawdown = portfolio.max_drawdown() * 100
win_rate = portfolio.win_rate() * 100

print(f"总收益率: {total_return:.2f}%")
print(f"年化收益率: {annual_return:.2f}%")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2f}%")
print(f"胜率: {win_rate:.2f}%")

# 7.2 交易统计
total_trades = portfolio.trades.count()
avg_trade_duration = portfolio.trades.duration.mean()

print(f"\n交易次数: {total_trades}")
print(f"平均持仓天数: {avg_trade_duration:.1f}")

# 7.3 获取基准（沪深 300 指数）
print("\n获取基准数据...")
benchmark = pro.index_daily(ts_code='399300.SZ', start_date=START_DATE, end_date=END_DATE)
benchmark['trade_date'] = pd.to_datetime(benchmark['trade_date'])
benchmark = benchmark.set_index('trade_date').sort_index()
benchmark = benchmark['close'].pct_change().fillna(0)

# 对齐日期
benchmark = benchmark.reindex(portfolio.returns().index, fill_value=0)

# 计算基准累计收益
benchmark_cumret = (1 + benchmark).cumprod()
benchmark_total_return = (benchmark_cumret.iloc[-1] - 1) * 100
benchmark_annual_return = ((benchmark_cumret.iloc[-1]) ** (252 / len(benchmark)) - 1) * 100

print(f"\n基准（沪深300）:")
print(f"总收益率: {benchmark_total_return:.2f}%")
print(f"年化收益率: {benchmark_annual_return:.2f}%")
print(f"超额收益: {total_return - benchmark_total_return:.2f}%")

# ============================================================================
# 8. 可视化
# ============================================================================

print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 8.1 净值曲线
portfolio_value = portfolio.value()
portfolio_value_norm = portfolio_value / portfolio_value.iloc[0]
benchmark_norm = benchmark_cumret / benchmark_cumret.iloc[0]

axes[0].plot(portfolio_value_norm.index, portfolio_value_norm.values,
             label='策略净值', linewidth=2)
axes[0].plot(benchmark_norm.index, benchmark_norm.values,
             label='沪深300', linewidth=2, alpha=0.7)
axes[0].set_title('策略净值曲线 vs 基准', fontsize=14, fontweight='bold')
axes[0].set_ylabel('净值（初始=1）', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 8.2 回撤曲线
drawdown = portfolio.drawdown() * 100
axes[1].fill_between(drawdown.index, 0, drawdown.values,
                      color='red', alpha=0.3, label='回撤')
axes[1].set_title('回撤曲线', fontsize=14, fontweight='bold')
axes[1].set_ylabel('回撤 (%)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# 8.3 月度收益热力图
monthly_returns = portfolio.returns().resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns_pivot = monthly_returns.to_frame('returns')
monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
monthly_returns_heatmap = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')

im = axes[2].imshow(monthly_returns_heatmap.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
axes[2].set_xticks(range(12))
axes[2].set_xticklabels(range(1, 13))
axes[2].set_yticks(range(len(monthly_returns_heatmap.index)))
axes[2].set_yticklabels(monthly_returns_heatmap.index)
axes[2].set_xlabel('月份', fontsize=12)
axes[2].set_ylabel('年份', fontsize=12)
axes[2].set_title('月度收益热力图 (%)', fontsize=14, fontweight='bold')

# 添加数值标签
for i in range(len(monthly_returns_heatmap.index)):
    for j in range(len(monthly_returns_heatmap.columns)):
        value = monthly_returns_heatmap.iloc[i, j]
        if not np.isnan(value):
            axes[2].text(j, i, f'{value:.1f}',
                        ha='center', va='center', fontsize=8)

plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig('/Users/huminghe/Documents/projects/quant-mh/backtest/reversal_20d_hs300_results.png',
            dpi=300, bbox_inches='tight')
print("图表已保存至: backtest/reversal_20d_hs300_results.png")

# ============================================================================
# 9. 因子 IC 检验
# ============================================================================

print("\n" + "=" * 60)
print("因子 IC 检验")
print("=" * 60)

# 计算因子 IC（信息系数）
# IC = 因子值与未来收益的相关系数
future_returns = close_prices.pct_change(21).shift(-21)  # 未来 21 日收益

ic_values = []
for date in rebalance_dates:
    if date not in returns_20d.index or date not in future_returns.index:
        continue

    factor = returns_20d.loc[date].dropna()
    future_ret = future_returns.loc[date].dropna()

    # 取交集
    common_stocks = factor.index.intersection(future_ret.index)
    if len(common_stocks) < 10:
        continue

    ic = factor[common_stocks].corr(future_ret[common_stocks])
    ic_values.append(ic)

ic_series = pd.Series(ic_values)
ic_mean = ic_series.mean()
ic_std = ic_series.std()
ic_ir = ic_mean / ic_std if ic_std > 0 else 0

print(f"IC 均值: {ic_mean:.4f}")
print(f"IC 标准差: {ic_std:.4f}")
print(f"IC_IR (信息比率): {ic_ir:.4f}")
print(f"IC > 0 的比例: {(ic_series > 0).mean() * 100:.1f}%")

# ============================================================================
# 10. 因子验证检查清单
# ============================================================================

print("\n" + "=" * 60)
print("因子验证检查清单")
print("=" * 60)

checks = {
    "IC 均值 > 0.03": ic_mean > 0.03,
    "IC_IR > 1.5": ic_ir > 1.5,
    "年化收益 > 15%": annual_return > 15,
    "最大回撤 < 30%": max_drawdown < 30,
    "夏普比率 > 1.0": sharpe_ratio > 1.0,
    "回测期 >= 5 年": len(close_prices) >= 252 * 5
}

for check, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"{status} {check}")

print("\n" + "=" * 60)
print("回测完成！")
print("=" * 60)
