## 文件职责边界

**IMPORTANT：每类内容只能存在于一个地方。添加内容前先确认归属，不要在多处重复定义。**

| 内容类型 | 唯一存放位置 | 禁止出现在 |
|----------|-------------|-----------|
| 全局行为规则（语言、回复风格、危险操作） | `~/.claude/CLAUDE.md` | 项目 CLAUDE.md |
| 项目概述、目录结构 | `CLAUDE.md`（本文件） | memory |
| A 股规范（技术栈、交易限制、策略方向） | `a_stock/CLAUDE.md` | 本文件 |
| 加密货币规范（执行流程、分析工作流） | `crypto/CLAUDE.md` | 本文件 |
| 交易成本、数据约定、回测标准 | `.claude/rules/trading-standards.md` | 本文件、lessons.md |
| 研究日志格式定义 | `crypto/docs/background/research_workflow.md` | 本文件（只引用） |
| 被纠正的错误和对应规则 | `.claude/lessons.md` | trading-standards.md |
| 跨会话的研究结论、项目状态 | `memory/` 各文件 | 本文件 |

**维护规则：**
- 修改某条规则时，搜索其他文件是否有相同内容，有则删除
- memory 条目描述的状态发生变化时，立即更新或删除，不要留过时快照
- 不要为"配置状态"创建 memory 条目
- **IMPORTANT：memory topic files 只存路径引用，不存实质内容。** 格式：一句话摘要 + `详见 crypto/docs/xxx.md`。有对应 docs 文件的 memory 条目，内容只能是指针，不能复制 docs 内容。这是防止 memory 和 docs 漂移的根本措施。
