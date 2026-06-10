# a_stock — CLAUDE.md

## 技术栈

- **数据源**：tushare（日线/财务/资金流）、akshare（免费补充数据）
- **交易接口**：待定（MiniQMT/XtQuant 或 XTP）
- **回测**：vectorbt（主力，多因子/ETF轮动）→ Qlib（因子积累后迁移）；期货用 vnpy 自带回测
- **部署**：本地 Mac 开发，实盘考虑云服务器（Linux 优先）

## 交易限制

交易成本、T+1、涨跌停、回测验收标准详见 `.claude/rules/trading-standards.md`。

- 无公开交易所 API，只能通过券商客户端接口
- A 股个股不适合趋势策略（T+1/无法做空/散户主导/政策市），商品期货 CTA 才是正确载体
- 因子选股时需过滤 ST 股和上市不足 6 个月的新股

## 策略方向

| 策略 | 仓位 | 接口 | 优先级 |
|------|------|------|--------|
| ETF 趋势轮动 | 30% | MiniQMT | 第一步，逻辑简单，验证框架 |
| 指数增强（多因子选股） | 30% | MiniQMT | 第二步，聚焦反转+质量因子 |
| 商品期货 CTA | 30% | CTP + vnpy | 可复用加密货币趋势策略经验 |
| 可转债双低 | 10% | MiniQMT | 降低整体波动，需过滤低评级 |

## 数据约定

数据复权方式、字段名、前视偏差处理详见 `.claude/rules/trading-standards.md`。

## 研究日志

研究过程日志追加到 `docs/research_log.md`（与 crypto 命名统一）。综合调研结论汇总在 `docs/research.md`。格式规范：`crypto/docs/research_workflow.md`。

## 参考文档

- 券商选型、数据源、交易接口：`docs/broker_and_cost.md`
- 因子计算要点、风险控制规则：`docs/strategy_notes.md`
- A 股量化调研结论汇总：`docs/research.md`
