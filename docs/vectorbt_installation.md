# VectorBT 安装指南

## 安装步骤

### 1. 安装 LLVM（正在进行中...）

```bash
brew install llvm
```

这一步可能需要 5-10 分钟，取决于网络速度和是否需要编译。

### 2. 配置 LLVM 环境变量

安装完成后需要配置环境变量：

```bash
# 添加到 ~/.zshrc
export LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

### 3. 安装 VectorBT

```bash
source venv/bin/activate
pip install vectorbt
```

### 4. 验证安装

```bash
python -c "import vectorbt as vbt; print(vbt.__version__)"
```

## 如果安装失败

### 方案 A：使用预编译的 llvmlite

```bash
pip install llvmlite  # 先单独安装
pip install vectorbt
```

### 方案 B：使用 backtrader 替代

VectorBT 功能强大但安装复杂，可以考虑使用更轻量的 backtrader：

```bash
pip install backtrader
```

### 方案 C：自己实现简单回测

不依赖任何回测框架，手动计算收益：

```python
# 简单的向量化回测
returns = close_prices.pct_change()
strategy_returns = (signals.shift(1) * returns).sum(axis=1)
cumulative_returns = (1 + strategy_returns).cumprod()
```

## 当前状态

- ⏳ LLVM 正在安装...
- ⏳ 等待 LLVM 安装完成
- ⏳ 然后安装 VectorBT

## 预计时间

- LLVM 安装：5-10 分钟
- VectorBT 安装：2-5 分钟
- 总计：7-15 分钟
