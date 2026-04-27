# VectorBT 安装问题与替代方案

## 问题分析

VectorBT 依赖 llvmlite，而 llvmlite 对 LLVM 版本要求严格：
- 我们安装了 LLVM 22.1.4
- 但 llvmlite 0.47 只支持 LLVM 14
- 版本不兼容导致编译失败

## 解决方案

### 方案 1：降级 LLVM（正在尝试）

```bash
brew uninstall llvm
brew install llvm@14
```

**问题：**
- 需要重新下载和编译
- 可能还会遇到其他兼容性问题
- 安装时间长

### 方案 2：使用 backtrader（推荐）

backtrader 是更轻量的回测框架，安装简单：

```bash
pip install backtrader
```

**优点：**
- 安装简单，无需 LLVM
- 文档完善
- 社区活跃
- 功能够用

**缺点：**
- 没有 VectorBT 那么快（向量化程度低）
- API 不如 VectorBT 简洁

### 方案 3：自己实现简单回测（最推荐）

对于你的项目，其实不需要复杂的回测框架：

```python
# 简单的向量化回测
import pandas as pd
import numpy as np

# 1. 计算策略收益
strategy_returns = (signals.shift(1) * returns).sum(axis=1)

# 2. 计算累计收益
cumulative_returns = (1 + strategy_returns).cumprod()

# 3. 计算性能指标
total_return = cumulative_returns.iloc[-1] - 1
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

# 4. 可视化
import matplotlib.pyplot as plt
cumulative_returns.plot()
plt.show()
```

**优点：**
- 完全控制，理解每一步
- 无依赖问题
- 运行快速
- 适合学习

**缺点：**
- 需要自己实现一些功能
- 没有现成的可视化

### 方案 4：使用在线平台

- 聚宽（免费）
- BigQuant（免费）
- 米筐（付费）

**优点：**
- 无需本地环境
- 数据和回测一体化
- 可视化完善

**缺点：**
- 代码在云端（有泄露风险）
- 受平台限制

## 我的建议

**对于你现在的阶段：**

### 短期（现在）：自己实现简单回测

```python
# 已经验证的部分
✅ 数据获取（Tushare）
✅ 因子计算（20 日收益率）
✅ 信号生成（买入最低 30%）

# 只需要补充
⏳ 收益计算（10 行代码）
⏳ 性能指标（20 行代码）
⏳ 可视化（10 行代码）
```

总共只需要 40 行代码，就能完成完整回测！

### 中期（1-2 周后）：使用 backtrader

等你熟悉了回测逻辑后，可以迁移到 backtrader：
- 更规范的框架
- 更完善的功能
- 无安装问题

### 长期（1 个月后）：评估是否需要 VectorBT

如果你的策略需要：
- 大规模参数优化
- 极致的性能
- 复杂的组合回测

那时再考虑解决 VectorBT 的安装问题。

## 立即可行的方案

让我帮你生成一个不依赖 VectorBT 的简单回测版本？

只需要：
- pandas
- numpy  
- matplotlib

这些都已经安装好了！

## 总结

**VectorBT 安装复杂度：⭐⭐⭐⭐⭐**
- 需要 LLVM
- 版本兼容问题
- 编译时间长

**自己实现简单回测：⭐**
- 40 行代码
- 无依赖问题
- 5 分钟完成

**建议：先用简单方案，快速验证策略逻辑，再考虑复杂框架。**
