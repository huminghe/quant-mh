# 加密货币策略研究日志

记录每次有实质结论的研究，格式见 CLAUDE.md 研究日志规范。

---

## 2026-05-25｜策略监控与研究闭环工具链搭建

**结论：** 完成三个工作流工具，形成"监控 → 分析 → 研究迭代"完整闭环。

**细节：**
- `health_report.py`：调用 Claude API 生成策略健康度报告，支持读最新结论 MD 或手动输入 --rr/--wr/--dd 指标；已集成到 `run_all.py --health`；当前评分 2-3/5（degrading），主因横盘磨损，未触发暂停
- `regime_monitor.py`：HMM 2状态市场制度监控，从 OKX 拉 BTC 日线（ccxt），特征为日收益率+20日滚动波动率；当前 RISK_ON 持续89天，近30天无 risk_off；HMM 实质上是低波动/高波动分类器（risk_off 波动率4.73% vs risk_on 2.04%）
- `backtest_analyst.py`（A股侧）：读回测 JSON → Claude 分析 → 输出归因+风险点+3个下一步假设；已在模拟数据上验证，假设质量可用

**下一步：** 配置 tushare token 后跑通 A 股反转策略回测，得到第一个真实研究闭环结果；加密货币侧先验证 OKX 多策略隔离问题，再推进 ATR 定仓和 KAMA。
