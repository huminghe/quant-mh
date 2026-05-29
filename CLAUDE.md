# quant-mh — CLAUDE.md

## 项目概述

量化交易研究与策略代码库，个人实盘项目，涵盖 A 股量化和加密货币策略两条线。

- 已有实盘经验：TradingView Pine Script 双向趋势策略 + Webhook → OKX/Binance（加密货币，约30个标的）
- 目标：将量化能力迁移到 A 股，研究数据获取、策略开发、回测、实盘执行全链路
- 资金量级：百万人民币级别

## 目录结构

```
quant-mh/
├── a_stock/           # A股量化（含 CLAUDE.md，进入时自动加载 A 股规范）
│   ├── strategies/
│   ├── backtest/
│   ├── data/
│   └── docs/
├── crypto/            # 加密货币策略（含 CLAUDE.md，进入时自动加载加密货币规范）
│   ├── analysis/      # run_all.py 一键入口
│   └── docs/
└── shared/
    └── scripts/
```

子项目规范：`a_stock/CLAUDE.md`（A 股技术栈、交易限制、策略方向）、`crypto/CLAUDE.md`（执行流程、分析工作流、研究日志）。

## 开发约定

- 策略文件命名：`{策略名}_{版本}.py`，如 `momentum_v1.py`
- A 股：数据脚本放 `a_stock/data/`，策略放 `a_stock/strategies/`，回测放 `a_stock/backtest/`
- 加密货币：分析脚本放 `crypto/analysis/`，文档放 `crypto/docs/`

交易成本、回测验收标准详见 `.claude/rules/trading-standards.md`（按需自动加载）。

## 范围控制

**IMPORTANT：只修改被明确要求修改的文件。** 不要顺手改"顺路看到的问题"，不要做未经要求的重构。发现不相关问题时提一句，等用户决定。

## 文件职责边界

@.claude/rules/file-boundaries.md

## 自动学习规则

**IMPORTANT：以下两条规则必须无条件执行，不需要用户提醒。**

1. **错误纠正 → 立即更新 lessons.md**：每当用户纠正你的行为，或你意识到自己犯了错，立即把对应规则追加到 `.claude/lessons.md`，格式：`- [类别]：[规则内容]`。

2. **compact 前 → 自动写 HANDOFF.md**：收到 compact 前的提示时，先把当前会话状态写入 `.claude/HANDOFF.md`（当前任务/已完成/下一步/关键约束/未解决问题），再执行 compact。

## 参考文档

- 券商选型、交易接口、交易成本：`a_stock/docs/broker_and_cost.md`
- 量化研究工作流规范：`crypto/docs/research_workflow.md`
- 历史错误与规则积累：`.claude/lessons.md`
