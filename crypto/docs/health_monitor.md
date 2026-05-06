# 策略健康度监控

最后更新：2026-05-06

## 监控频率

每月月底执行一次，计算过去3个月所有已平仓交易的统计数据。

## 监控指标

| 指标 | 历史基准 | 预警阈值 | 说明 |
|------|----------|----------|------|
| 滚动3个月盈亏比 | 2.26 | < 1.58（-30%） | 最核心指标 |
| 滚动3个月胜率 | 37.5% | < 30%（-20%） | 辅助指标 |
| 最大回撤 / 历史最大回撤 | 1.0x | > 1.5x | 触发正式审查 |

## 触发规则

| 情况 | 动作 |
|------|------|
| 单月盈亏比低于1.58 | 记录，继续执行，不调整 |
| 连续2个月盈亏比低于1.58 | 触发审查 |
| 最大回撤超历史最大回撤1.5倍 | 暂停策略，重新评估 |
| 以上都未触发 | 当前是噪声，不需要行动 |

## 审查流程

触发审查时，先做归因再决定是否调整：

1. **消息面冲击？** 把异常大亏损（超过正常止损1.5倍）剔除，重新计算盈亏比。如果剔除后回到正常范围，是外部冲击，不是策略问题。

2. **横盘磨损？** 看当前市场是否处于低波动横盘状态（ADX < 20，布林带收窄）。如果是，等待市场方向确立，暂停机制是主要应对手段。

3. **策略本身失效？** 排除以上两种情况后，盈亏比仍然持续低位，才考虑策略参数审查。

## 心态管理原则

- 焦虑时对照这张表，指标没触发就是噪声
- 单次不符合预期不触发调整，需要连续2次
- 胜率37%的策略，连续8次亏损概率2.5%，属于正常范围
- 要判断策略真正失效，需要至少50-100笔交易的滚动样本

## 历史参照

历史上盈亏比低谷后均回归均值：

| 时期 | 季度盈亏比 | 之后表现 |
|------|-----------|----------|
| 2021Q3 | 1.35 | 2021Q4反弹至2.02 |
| 2022Q1 | 1.40 | 2022Q2反弹至3.53 |
| 2023Q3 | 1.73 | 2023Q4反弹至2.59 |
| 2026Q2 | 1.14 | 待观察 |

## 计算工具

使用 `crypto/analysis/strategy_analysis.py` 分析导出的 CSV：

```bash
source venv/bin/activate
python crypto/analysis/strategy_analysis.py ~/Downloads/策略导出文件.csv
```

滚动盈亏比也可以在 TradingView 策略里直接画，参考 Pine Script v6 实现：

```pine
N = input.int(30, "滚动窗口（笔数）")
total_trades = strategy.closedtrades
start_idx = math.max(0, total_trades - N)
float sum_profit = 0.0
float sum_loss = 0.0
int count_profit = 0
int count_loss = 0
for i = start_idx to total_trades - 1
    pnl = strategy.closedtrades.profit(i)
    if pnl > 0
        sum_profit += pnl
        count_profit += 1
    else if pnl < 0
        sum_loss += math.abs(pnl)
        count_loss += 1
avg_profit = count_profit > 0 ? sum_profit / count_profit : 0.0
avg_loss   = count_loss  > 0 ? sum_loss  / count_loss  : 0.0
rolling_rr = avg_loss > 0 ? avg_profit / avg_loss : 0.0
plot(rolling_rr, "滚动盈亏比", color=color.blue, linewidth=2)
hline(2.26, "历史均值", color=color.green)
hline(1.58, "预警线", color=color.red, linestyle=hline.style_dashed)
```
