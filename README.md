# quant-mh

量化交易研究与策略代码库，涵盖 A 股量化和加密货币策略。

## 目录结构

- `a_stock/` — A 股量化（主要研究方向）
  - `strategies/` — 交易策略
  - `backtest/` — 回测脚本与结果
  - `data/` — 数据获取与处理
  - `docs/` — 调研笔记
- `crypto/` — 加密货币策略
  - `analysis/` — 策略分析脚本
  - `docs/` — 调研笔记与设计文档
- `shared/` — 共用工具
  - `scripts/` — RSS、数据脚本等

## 技术栈

- 语言：Python
- A 股数据源：tushare + akshare
- A 股交易接口：MiniQMT (XtQuant) / XTP
- A 股回测：vectorbt → Qlib
- 加密货币：TradingView Pine Script + Webhook → OKX/Binance

