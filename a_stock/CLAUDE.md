# a_stock — CLAUDE.md

## 技术栈

- **数据源**：tushare（日线/财务/资金流）、akshare（免费补充数据）
- **交易接口**：待定（MiniQMT/XtQuant 或 XTP）→ 详见 `docs/broker_and_cost.md`
- **回测**：vectorbt（主力，多因子/ETF轮动）→ Qlib（因子积累后迁移）；期货用 vnpy 自带回测
- **部署**：本地 Mac 开发，实盘考虑云服务器（Linux 优先）

## 交易限制

- 无公开交易所 API，只能通过券商客户端接口
- T+1：当日买入不可当日卖出，回测信号需延迟一天执行
- 涨跌停：涨停无法买入、跌停无法卖出，回测中必须模拟无法成交
- 高频策略受印花税（千1，仅卖出）严重压制，个人量化更适合中低频
- A 股个股不适合趋势策略（T+1/无法做空/散户主导/政策市），商品期货 CTA 才是正确载体
- 因子选股时需过滤 ST 股和上市不足 6 个月的新股

## 策略方向

| 策略 | 仓位 | 接口 | 优先级 |
|------|------|------|--------|
| ETF 趋势轮动 | 30% | MiniQMT | 第一步，逻辑简单，验证框架 |
| 指数增强（多因子选股） | 30% | MiniQMT | 第二步，聚焦反转+质量因子 |
| 商品期货 CTA | 30% | CTP + vnpy | 可复用加密货币趋势策略经验 |
| 可转债双低 | 10% | MiniQMT | 降低整体波动，需过滤低评级 |

详细策略笔记：`docs/strategy_notes.md`

## 数据约定

- **复权**：默认使用后复权（`adj_type='hfq'`），因子计算用后复权价格
- **tushare 日线字段**：`trade_date`、`open/high/low/close/vol/amount`
- **akshare 字段**：日期列名可能是"日期"或"date"，使用前统一重命名
- **前视偏差**：财务数据季度披露，需用 point-in-time 数据，安全缓冲期 45-60 天

## 研究日志

研究结论追加到 `docs/factor_research_log.md`，使用根级 `crypto/docs/research_workflow.md` 中定义的完整格式。
