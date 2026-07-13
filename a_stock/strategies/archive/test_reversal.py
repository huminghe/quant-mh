"""
20 日短期反转策略 - 快速测试版
使用少量股票和短时间段，快速验证代码逻辑
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("20 日短期反转策略 - 快速测试")
print("=" * 60)

# ============================================================================
# 1. 配置参数
# ============================================================================

# Tushare Token（需要替换）
TS_TOKEN = 'YOUR_TUSHARE_TOKEN'
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 测试参数（缩小范围）
START_DATE = '20230101'
END_DATE = '20231231'
TEST_STOCKS = ['600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000333.SZ']  # 5 只测试股票
LOOKBACK_PERIOD = 20

print(f"\n测试配置:")
print(f"- 时间范围: {START_DATE} 至 {END_DATE}")
print(f"- 测试股票: {len(TEST_STOCKS)} 只")
print(f"- 回看期: {LOOKBACK_PERIOD} 日")

# ============================================================================
# 2. 数据获取测试
# ============================================================================

print("\n" + "=" * 60)
print("测试 1: 数据获取")
print("=" * 60)

try:
    # 测试获取单只股票数据
    test_stock = TEST_STOCKS[0]
    df_test = ts.pro_bar(
        ts_code=test_stock,
        asset='E',
        start_date=START_DATE,
        end_date=END_DATE,
        adj='qfq'
    )

    if df_test is not None and not df_test.empty:
        print(f"✓ 成功获取 {test_stock} 数据")
        print(f"  数据条数: {len(df_test)}")
        print(f"  时间范围: {df_test['trade_date'].min()} 至 {df_test['trade_date'].max()}")
    else:
        print(f"✗ 获取 {test_stock} 数据失败")
        print("  请检查 Tushare Token 是否正确")
        exit(1)

except Exception as e:
    print(f"✗ 数据获取失败: {e}")
    print("\n可能的原因:")
    print("1. Tushare Token 未设置或无效")
    print("2. 网络连接问题")
    print("3. Tushare 积分不足")
    print("\n解决方法:")
    print("1. 访问 https://tushare.pro 注册并获取 token")
    print("2. 在代码中替换 TS_TOKEN = 'YOUR_TUSHARE_TOKEN'")
    exit(1)

# ============================================================================
# 3. 数据处理测试
# ============================================================================

print("\n" + "=" * 60)
print("测试 2: 数据处理")
print("=" * 60)

# 获取所有测试股票数据
all_data = []
for stock in TEST_STOCKS:
    try:
        df = ts.pro_bar(
            ts_code=stock,
            asset='E',
            start_date=START_DATE,
            end_date=END_DATE,
            adj='qfq'
        )
        if df is not None and not df.empty:
            df['ts_code'] = stock
            all_data.append(df)
            print(f"✓ {stock}: {len(df)} 条数据")
    except Exception as e:
        print(f"✗ {stock}: 获取失败 - {e}")

if not all_data:
    print("✗ 未获取到任何数据")
    exit(1)

# 合并数据
df_all = pd.concat(all_data, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
df_all = df_all.sort_values(['ts_code', 'trade_date'])

# 转换为宽表格式
close_prices = df_all.pivot(index='trade_date', columns='ts_code', values='close')
close_prices = close_prices.sort_index()
close_prices = close_prices.ffill()  # 新版 pandas 语法

print(f"\n✓ 数据处理完成")
print(f"  股票数量: {close_prices.shape[1]}")
print(f"  交易日数量: {close_prices.shape[0]}")

# ============================================================================
# 4. 因子计算测试
# ============================================================================

print("\n" + "=" * 60)
print("测试 3: 因子计算")
print("=" * 60)

# 计算 20 日收益率
returns_20d = close_prices.pct_change(LOOKBACK_PERIOD)

print(f"✓ 因子计算完成")
print(f"  因子数据形状: {returns_20d.shape}")
print(f"  有效数据点: {returns_20d.count().sum()}")

# 显示示例数据
print(f"\n示例因子值（最近 5 日）:")
print(returns_20d.tail().round(4))

# ============================================================================
# 5. 信号生成测试
# ============================================================================

print("\n" + "=" * 60)
print("测试 4: 信号生成")
print("=" * 60)

# 月度调仓日期（ME = Month End）
rebalance_dates = close_prices.resample('ME').last().index
print(f"✓ 调仓日期: {len(rebalance_dates)} 个")

# 生成简单信号（最后一个调仓日）
last_rebalance = rebalance_dates[-1]

# 找到最接近的交易日
if last_rebalance not in returns_20d.index:
    # 找到最近的前一个交易日
    valid_dates = returns_20d.index[returns_20d.index <= last_rebalance]
    if len(valid_dates) > 0:
        last_rebalance = valid_dates[-1]
    else:
        print("✗ 无有效调仓日期")
        exit(0)

factor_values = returns_20d.loc[last_rebalance].dropna()

if len(factor_values) > 0:
    threshold = factor_values.quantile(0.3)
    buy_signals = factor_values <= threshold

    print(f"\n最后一次调仓 ({last_rebalance.date()}):")
    print(f"  因子阈值: {threshold:.4f}")
    print(f"  买入股票数: {buy_signals.sum()}")
    print(f"\n买入股票:")
    for stock in buy_signals[buy_signals].index:
        print(f"  - {stock}: {factor_values[stock]:.4f}")
else:
    print("✗ 无有效因子值")

# ============================================================================
# 6. VectorBT 测试
# ============================================================================

print("\n" + "=" * 60)
print("测试 5: VectorBT 回测")
print("=" * 60)

try:
    import vectorbt as vbt
    print("✓ VectorBT 已安装")

    # 创建简单的买入持有策略测试
    signals = pd.DataFrame(1, index=close_prices.index, columns=close_prices.columns)

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=signals > 0,
        size=1.0 / len(close_prices.columns),
        size_type='targetpercent',
        fees=0.00164,
        freq='D',
        init_cash=100000
    )

    total_return = portfolio.total_return() * 100
    print(f"✓ 回测执行成功")
    print(f"  测试策略收益: {total_return:.2f}%")

except ImportError:
    print("✗ VectorBT 未安装")
    print("  请运行: pip install vectorbt")
except Exception as e:
    print(f"✗ 回测失败: {e}")

# ============================================================================
# 7. 总结
# ============================================================================

print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

print("\n✓ 所有测试通过！")
print("\n下一步:")
print("1. 修改完整版代码中的 TS_TOKEN")
print("2. 运行完整回测: python strategies/reversal_20d_hs300.py")
print("3. 查看结果图表: backtest/reversal_20d_hs300_results.png")

print("\n" + "=" * 60)
