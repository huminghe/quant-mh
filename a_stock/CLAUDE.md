# a_stock — CLAUDE.md

## 技术栈

- **数据源**：tushare（日线/财务/资金流）、akshare（免费补充数据）
- **交易接口**：待定（MiniQMT/XtQuant 或 XTP）
- **回测**：自实现 pandas/numpy 回测（vectorbt 因 LLVM 依赖冲突未装成，实际全部脚本走此方案）→ Qlib（因子积累后迁移）；期货用 vnpy 自带回测
- **部署**：本地 Mac 开发，实盘考虑云服务器（Linux 优先）

## 交易限制

交易成本、T+1、涨跌停、回测验收标准详见 `.claude/rules/trading-standards.md`。

- 无公开交易所 API，只能通过券商客户端接口
- A 股个股不适合趋势策略（T+1/无法做空/散户主导/政策市），商品期货 CTA 才是正确载体
- 因子选股时需过滤 ST 股和上市不足 6 个月的新股

## 策略方向

**这张表是 A 股四条策略线的总览入口，一句话状态+指向详情文档。IMPORTANT：每次某条策略线产出阶段性结论（新方向验证完成/候选通过或证伪/接入实盘等）后，必须回来同步对应这一行，不能只更新 research 文档或 memory——这是过去实际发生过的疏漏（第四十六轮候选进展写了 research 文档和 memory，忘了同步这里）。**

| 策略 | 仓位 | 接口 | 优先级 |
|------|------|------|--------|
| ETF 趋势轮动 | 30% | MiniQMT | 45只手工标的池模拟盘已暂停并放弃；机械化431池+flow信号模拟盘监控中（未真实下单），候选池优化方向已穷尽（详细进度见 memory） |
| 指数增强（多因子选股） | 30% | MiniQMT | 因子选股方向遇到较大困难；指数样本股调整效应（事件驱动）已通过完整验证待接入；第四十六轮候选（残差动量resmom沪深300/异常换手率atr中证500）IC初筛+组合消融+滚动稳健性检验均通过，仍需补窗口敏感性复核+DSR多重检验校正才可接入实盘（2026-09-23，详细进度见 memory 和 `docs/research_index_enhancement.md`） |
| 商品期货 CTA | 30% | CTP + vnpy | 可复用加密货币趋势策略经验，下一开发方向 |
| 可转债双低 | - | MiniQMT | 已证伪，不接入实盘：IC初筛未过筛+信用过滤无效+样本外过拟合（详见 memory/docs） |

## 数据约定

数据复权方式、字段名、前视偏差处理详见 `.claude/rules/trading-standards.md`。

## 研究日志

研究过程和结论按策略线拆分记录：ETF 轮动追加到 `docs/research_etf_rotation.md`，指数增强追加到 `docs/research_index_enhancement.md`，可转债追加到 `docs/research_convertible_bond.md`；跨策略/跨项目通用内容留在 `docs/research.md`。格式规范参考 `crypto/docs/background/research_workflow.md`（数据范围/测试规模/参数选择理由/过拟合风险等要素）。

## 参考文档

- 券商选型、数据源、交易接口：`docs/broker_and_cost.md`
- 跨策略通用的因子计算要点、回测工具选型：`docs/strategy_notes.md`
- ETF 轮动策略细节：`docs/ETF轮动调研.md`；调研日志：`docs/research_etf_rotation.md`；实盘/模拟盘表现记录：`docs/ETF轮动实盘_模拟盘记录.md`
- 多因子选股（指数增强）策略细节（含风险控制规则）：`docs/多因子选股调研.md`；调研日志：`docs/research_index_enhancement.md`
- 可转债双低策略调研（已证伪）：`docs/research_convertible_bond.md`
- 跨策略/跨项目通用调研结论：`docs/research.md`
