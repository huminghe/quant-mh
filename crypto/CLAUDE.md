# crypto — CLAUDE.md

## 策略执行

- **策略**：双向趋势跟踪（做多+做空），Pine Script v6，约 30 个标的
- **执行**：TradingView Webhook → OKX/Binance 全自动
- **市场数据**：调研价格、代币信息时必须先用网络搜索（Exa），不能依赖训练数据

## 分析工作流

一键入口：`analysis/run_all.py`，自动扫描 xlsx → 生成 MD 结论 + PNG 图表 + HTML 交互图表 + Excel 报告；支持 `--llm` 自动生成 Claude 解读。

分析脚本保存到 `analysis/`，不只放 `/tmp`。命名规范：`{主题}_{功能}.py`。

**目录分工：**
- `analysis/`（根目录）：只放活跃脚本（常用入口、核心库、仍在迭代的分析）
- `analysis/archive/`：已完结的验证脚本（结论已固化进 memory 或 research_log），保留不删除
- `analysis/reports/`：所有生成产物（`charts_*/`、`health_report_*.md`、`regime_monitor_*`），不提交 git

**判断"已完结"的标准：** 该脚本验证的结论已写入 `docs/strategy_research_log.md` 或 `docs/filters_validation.md`，且不需要再次运行。

## 研究日志

研究结论追加到 `docs/strategy_research_log.md`，**IMPORTANT：使用 `docs/research_workflow.md` 中定义的完整格式**（含数据范围、测试规模、参数选择理由、过拟合风险、样本外验证、排除的方向）。

## 研究总结检查清单

用户说"总结"或"更新文档"时，按顺序执行以下4项，全部完成后再汇报：

1. **保存 /tmp 脚本**：检查 `/tmp/*.py`，把有独立分析价值的最终版本复制到 `analysis/`，加上结论性注释头。中间调试版本（命名带 debug/entry2/fix 等）不用保存。
2. **写研究日志**：有实质结论的研究追加到 `docs/strategy_research_log.md`。
3. **标注已有文件的局限性**：如果本次研究发现某个已有脚本或文档有问题，在原文件里直接标注，不能只记在日志里。
4. **更新"明确不做"列表**：如果得出"这个方向不值得做"的结论，追加到 `project_crypto_optimization_roadmap.md` 的"明确不做"表格。

## 文档组织原则

**按稳定程度分文件，不按研究深度分文件：**
- 框架类（理论、改进方向，很少改动）→ 独立文档，如 `trend_following_research.md`
- 实验流水账（过程、数据、初步发现，持续追加）→ 研究日志
- 高频查阅的结论索引（测了 10+ 个指标、需要快速查参数）→ 独立索引文档，如 `filters_validation.md`
- 已完结/已过期的文档 → `docs/archive/`

**判断"已完结"的标准：** 该文档描述的决策已执行、结论已固化进其他文档，或已被更新版本取代（顶部有"已由 X 取代"标注）。

**单个指标验证结论追加到研究日志即可，不需要单独建文档。**

## 回撤计算

使用固定资本分母（如每标的 20000 USDT），不用峰值百分比；计算前先说明方法。

## 参考文档

- 暂停机制：`docs/pause_mechanism.md`
- 健康度监控：`docs/health_monitor.md`
- 趋势跟踪研究框架：`docs/trend_following_research.md`
- 过滤器验证结论索引：`docs/filters_validation.md`
- 研究工作流规范：`docs/research_workflow.md`
