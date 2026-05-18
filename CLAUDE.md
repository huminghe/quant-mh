# quant-mh — CLAUDE.md

## 项目概述

量化交易研究与策略代码库，个人实盘项目，涵盖 A 股量化和加密货币策略两条线。

## 背景与现状

- 已有实盘经验：TradingView Pine Script 双向趋势策略 + Webhook → OKX/Binance（加密货币，约30个标的）
- 目标：将量化能力迁移到 A 股，研究数据获取、策略开发、回测、实盘执行全链路
- 资金量级：百万人民币级别

## 技术栈

### A 股
- **语言**：Python
- **数据源**：tushare（日线/财务/资金流）、akshare（免费补充数据）
- **交易接口**：待定（MiniQMT/XtQuant 或 XTP）→ 详见 @a_stock/docs/broker_and_cost.md
- **回测**：vectorbt（主力，多因子/ETF轮动）→ Qlib（因子积累后迁移）；期货用 vnpy 自带回测
- **部署**：本地 Mac 开发，实盘考虑云服务器（Linux 优先）

### 加密货币
- **策略**：双向趋势跟踪（做多+做空），Pine Script v6，约30个标的
- **执行**：TradingView Webhook → OKX/Binance 全自动
- **分析**：Python（`crypto/analysis/run_analysis.py` 一键入口，自动扫描 xlsx → 生成 HTML 图表 + Excel 报告）
- **文档**：暂停机制（`crypto/docs/pause_mechanism.md`）、健康度监控（`crypto/docs/health_monitor.md`）、趋势跟踪研究（`crypto/docs/trend_following_research.md`）

## 目录结构

```
quant-mh/
├── CLAUDE.md          # 本文件
├── README.md
├── a_stock/           # A股量化（主要研究方向）
│   ├── strategies/    # 交易策略
│   ├── backtest/      # 回测脚本与结果
│   ├── data/          # 数据获取与处理脚本
│   └── docs/          # A股相关调研笔记
├── crypto/            # 加密货币策略
│   ├── analysis/      # 策略分析脚本（run_analysis.py 一键入口，含收益/回撤/相关性/热力图/滚动分析）
│   └── docs/          # 调研笔记、暂停机制设计文档
└── shared/            # 两个项目共用
    └── scripts/       # RSS工具、数据脚本等
```

## A股交易限制（重要背景）

- 无公开交易所 API，个人只能通过券商客户端接口
- T+1 限制，涨跌停板，需在策略中特殊处理
- 高频策略受印花税（千1，仅卖出）严重压制，个人量化更适合中低频
- A 股个股不适合趋势策略（T+1/无法做空/散户主导/政策市），商品期货 CTA 才是正确载体

## 开发约定

- 代码注释语言：中文
- 策略文件命名：`{策略名}_{版本}.py`，如 `momentum_v1.py`
- A 股：数据脚本放 `a_stock/data/`，策略放 `a_stock/strategies/`，回测放 `a_stock/backtest/`
- 加密货币：分析脚本放 `crypto/analysis/`，文档放 `crypto/docs/`
- 敏感信息（token、账户）统一用环境变量，不写入代码

## 策略方向

| 策略 | 仓位 | 接口 | 优先级 |
|------|------|------|--------|
| ETF 趋势轮动 | 30% | MiniQMT | 第一步，逻辑简单，验证框架 |
| 指数增强（多因子选股） | 30% | MiniQMT | 第二步，聚焦反转+质量因子 |
| 商品期货 CTA | 30% | CTP + vnpy | 可复用加密货币趋势策略经验 |
| 可转债双低 | 10% | MiniQMT | 降低整体波动，需过滤低评级 |

**选择依据：** 趋势经验迁移到期货CTA；ETF轮动替代A股个股趋势；百万资金不适合市场中性，指数增强更合适；CTA与股票低相关，熊市天然对冲。

详细策略笔记：@a_stock/docs/strategy_notes.md

## 参考文档

- 券商选型、数据源、交易接口、交易成本：@a_stock/docs/broker_and_cost.md
- 多因子/ETF轮动/风控详细笔记：@a_stock/docs/strategy_notes.md
