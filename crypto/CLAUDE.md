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

**判断"已完结"的标准：** 该脚本验证的结论已写入 `docs/active/strategy_research_log.md` 或 `docs/active/validation_results.md`，且不需要再次运行。

## 研究日志

研究结论追加到 `docs/active/strategy_research_log.md`，**IMPORTANT：使用 `docs/background/research_workflow.md` 中定义的完整格式**（含数据范围、测试规模、参数选择理由、过拟合风险、样本外验证、排除的方向）。

## 研究总结检查清单

用户说"总结"或"更新文档"时，按顺序执行以下5项，全部完成后再汇报：

1. **保存 /tmp 脚本**：检查 `/tmp/*.py`，把有独立分析价值的最终版本复制到 `analysis/`，加上结论性注释头。中间调试版本（命名带 debug/entry2/fix 等）不用保存。
2. **写研究日志**：有实质结论的研究追加到 `docs/active/strategy_research_log.md`。
3. **标注已有文件的局限性**：如果本次研究发现某个已有脚本或文档有问题，在原文件里直接标注，不能只记在日志里。
4. **更新"明确不做"列表**：如果得出"这个方向不值得做"的结论，追加到 `docs/active/validation_results.md` 的快速索引表。
5. **更新 memory 文件**：如果有新的结论或决策，更新对应的 memory 文件（有 docs 对应的只更新指针摘要，无对应的更新实质内容）。memory 路径：`~/.claude/projects/-Users-huminghe-Documents-projects-quant-mh/memory/`，优化进展对应 `project_crypto_optimization_roadmap.md`。

## 文档组织原则

**按稳定程度分文件，不按研究深度分文件：**
- 框架类（理论、改进方向，很少改动）→ `docs/background/`
- 实验流水账（过程、数据、初步发现，持续追加）→ `docs/active/strategy_research_log.md`
- 高频查阅的结论索引（测了 10+ 个指标、需要快速查参数）→ `docs/active/validation_results.md`
- 已过期/被取代的文档 → `docs/archive/`

**归档标准（同时满足以下两条才能归档）：**
1. 内容已被其他文档完全吸收（新文档包含原文档的所有有效信息）
2. 原文档不再有独立参考价值（删掉不会造成信息丢失）

**不能作为归档理由：** "决策已执行"。执行了的决策文档恰恰需要保留——它记录的是"为什么这样做"，是当前配置的依据。

**单个指标验证结论追加到研究日志即可，不需要单独建文档。**

## 数据合并前置检查

**⚠️ CRITICAL：用外部数据与 TV 交易记录合并前，必须先对齐时区。此错误曾导致 ER/CI 所有结论全部作废。**

- TV 导出的交易记录是 UTC+8 naive datetime
- Binance/OKX API 返回 UTC 时间戳
- 直接 merge_asof 等效于每次用了未来 8 小时数据（4 根 2H bar），导致所有正向结论都是假的
- 修正方法：`df['entry_dt'] = pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)`
- 验证方法：取一笔已知入场时间的交易，检查合并后匹配到的 K 线时间是否合理

**⚠️ TV xlsx 导出格式变更（2026-06-10）：**
- 交易记录 sheet 名从 `交易清单` 改成 `交易`
- 分析脚本已兼容，回退顺序：`交易清单` → `交易` → `List of trades` → `Trades`
- 字段名变更：盈亏比 `平均胜率/平均负率` → `平均盈利/平均亏损`；持仓K线 `交易的平均#K线数` → `交易者平均K线`
- 已修复文件：`analysis_utils.py`、`report_md.py`、`analyze_strategies.py`、`plot_equity_curves.py`

## 回撤计算

使用固定资本分母（如每标的 20000 USDT），不用峰值百分比；计算前先说明方法。

## 策略版本命名规范

| 标的 | 文件命名 | 时间框架 | 说明 |
|------|---------|---------|------|
| ETH | ETH_ema | 479m | 基础 EMA 策略 |
| ETH | ETH_v2 | 479m | 改进版，夏普最高 |
| ETH | ETH_v3_205m | 205m | 205分钟，**只跑这个，不跑 v3_3h**（两者相关性 0.935） |
| ETH | ETH_v3_3h | 3h | 与 v3_205m 高度相关，不同时运行 |
| SOL | SOL_ema | 479m | 基础版本 |
| SOL | SOL_v2 | 479m | 改进趋势，夏普最高 |
| SOL | SOL_v3_205m | 205m | 与其他策略零相关，核心策略 |
| SOL | SOL_v5_3h | 3h | 与 ema/v2 中度相关 |

**注意：** SOL 的 v3 是 205m（不是 3h），与 ETH 版本命名不对应。历史记录中曾称 205m 版本为"V4"，当前脚本统一命名为 `v3_205m`。

## 参考文档

**active/**（当前决策依据，保持最新）
- 暂停机制：`docs/active/pause_mechanism.md`
- 健康度监控：`docs/active/health_monitor.md`
- 验证结论索引（高频查阅，保持同步）：`docs/active/validation_results.md`
- 策略版本性能对比与当前部署配置：`docs/active/strategy_versions.md`
- 多标的多策略组合分析（研究过程）：`docs/active/multi_asset_combination_analysis.md`
- 代币选择框架：`docs/active/token_selection_framework.md`
- 研究日志：`docs/active/strategy_research_log.md`

**background/**（背景理解，偶尔查阅）
- 趋势跟踪研究框架：`docs/background/trend_following_research.md`
- 研究工作流规范：`docs/background/research_workflow.md`
- 交易量数据可靠性调研：`docs/background/volume_research.md`

**archive/**（结论已固化）：`docs/archive/`

**其他**
- 历史错误与规则积累：`.claude/lessons.md`
- 交易成本、数据约定、回测验收标准：`.claude/rules/trading-standards.md`
