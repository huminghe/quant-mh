"""
Tushare 连接测试
"""

import tushare as ts
import os

print("=" * 60)
print("Tushare 连接测试")
print("=" * 60)

# 方式 1：从环境变量读取（推荐）
token = os.getenv('TUSHARE_TOKEN')

# 方式 2：直接写在代码中（不推荐，容易泄露）
if not token:
    token = 'YOUR_TOKEN_HERE'  # 替换为你的 Token

if not token or token == 'YOUR_TOKEN_HERE':
    print("\n❌ Token 未配置")
    print("\n请选择以下方式之一配置 Token：")
    print("\n方式 1（推荐）：")
    print("1. 编辑 ~/.zshrc")
    print("2. 添加: export TUSHARE_TOKEN='你的Token'")
    print("3. 运行: source ~/.zshrc")
    print("\n方式 2：")
    print("1. 修改本文件")
    print("2. 替换 token = 'YOUR_TOKEN_HERE' 为你的实际 Token")
    print("\n获取 Token：https://tushare.pro/user/token")
    exit(1)

print(f"\n✓ Token 已配置: {token[:10]}...{token[-10:]}")

# 设置 Token
ts.set_token(token)
pro = ts.pro_api()

# 测试 1：获取交易日历
print("\n" + "=" * 60)
print("测试 1: 获取交易日历")
print("=" * 60)

try:
    df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240131')
    if df is not None and not df.empty:
        print(f"✓ 成功获取交易日历")
        print(f"  数据条数: {len(df)}")
        trading_days = df[df['is_open'] == 1]
        print(f"  2024年1月交易日: {len(trading_days)} 天")
    else:
        print("✗ 获取交易日历失败")
except Exception as e:
    print(f"✗ 获取交易日历失败: {e}")
    exit(1)

# 测试 2：获取股票基本信息
print("\n" + "=" * 60)
print("测试 2: 获取股票基本信息")
print("=" * 60)

try:
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    if df is not None and not df.empty:
        print(f"✓ 成功获取股票列表")
        print(f"  上市股票数量: {len(df)}")
        print(f"\n示例数据（前 5 只）:")
        print(df.head().to_string(index=False))
    else:
        print("✗ 获取股票列表失败")
except Exception as e:
    print(f"✗ 获取股票列表失败: {e}")
    exit(1)

# 测试 3：获取股票日线数据
print("\n" + "=" * 60)
print("测试 3: 获取股票日线数据")
print("=" * 60)

try:
    # 获取贵州茅台最近 5 天数据
    df = ts.pro_bar(ts_code='600519.SH', adj='qfq', start_date='20240101', end_date='20240131')
    if df is not None and not df.empty:
        print(f"✓ 成功获取日线数据")
        print(f"  股票: 600519.SH (贵州茅台)")
        print(f"  数据条数: {len(df)}")
        print(f"\n最近 5 天数据:")
        print(df[['trade_date', 'open', 'high', 'low', 'close', 'vol']].head().to_string(index=False))
    else:
        print("✗ 获取日线数据失败")
except Exception as e:
    print(f"✗ 获取日线数据失败: {e}")
    print("\n可能的原因:")
    print("1. Tushare 积分不足（需要 120 积分才能获取日线数据）")
    print("2. 访问频率过快（免费用户有限流）")
    print("\n解决方法:")
    print("1. 访问 https://tushare.pro/document/1?doc_id=13 查看积分规则")
    print("2. 完成新手任务获取积分")
    print("3. 等待几秒后重试")

# 测试 4：检查积分
print("\n" + "=" * 60)
print("测试 4: 检查积分")
print("=" * 60)

try:
    # 获取用户信息（需要积分）
    df = pro.user()
    if df is not None and not df.empty:
        points = df.iloc[0]['points']
        print(f"✓ 当前积分: {points}")

        print(f"\n积分说明:")
        print(f"  - 注册送 100 积分")
        print(f"  - 完成新手任务可获得更多积分")
        print(f"  - 日线数据需要 120 积分")
        print(f"  - 财务数据需要 2000 积分")

        if points < 120:
            print(f"\n⚠️  当前积分不足，无法获取日线数据")
            print(f"  请访问 https://tushare.pro/document/1?doc_id=13 完成任务")
        else:
            print(f"\n✓ 积分充足，可以正常使用")
    else:
        print("✗ 无法获取用户信息")
except Exception as e:
    print(f"✗ 获取用户信息失败: {e}")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

print("\n✓ Tushare 连接成功！")
print("\n下一步:")
print("1. 如果积分不足，完成新手任务: https://tushare.pro/document/1?doc_id=13")
print("2. 运行策略测试: python strategies/test_reversal.py")
print("3. 运行完整回测: python strategies/reversal_20d_hs300.py")

print("\n" + "=" * 60)
