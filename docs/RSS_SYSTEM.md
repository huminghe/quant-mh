# RSS 订阅系统说明

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     RSS MCP 系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  RSS 源      │      │  RSS 源      │                    │
│  │  (arXiv)     │      │ (Quantpedia) │  ...               │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │   RSS MCP Server    │                            │
│         │  (Node.js 项目)      │                            │
│         │  - 抓取 RSS          │                            │
│         │  - 解析内容          │                            │
│         │  - SQLite 存储       │                            │
│         │  - 26 个工具         │                            │
│         └──────────┬──────────┘                            │
│                    │                                        │
│         ┌──────────▼──────────┐                            │
│         │   使用接口           │                            │
│         ├─────────────────────┤                            │
│         │ 1. Shell 脚本        │ ← 推荐日常使用              │
│         │ 2. Python 脚本       │ ← 编程集成                 │
│         │ 3. 原始命令          │ ← 完整功能                 │
│         └─────────────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心概念

### 1. RSS MCP 是什么？

- **独立的 Node.js 项目**，不依赖 Claude
- 位置：`/Users/huminghe/.mcp-servers/rss-mcp`
- 功能：RSS 订阅管理、文章抓取、智能分析
- 数据：存储在 SQLite 数据库（`data/rss.db`）

### 2. MCP 协议

- **Model Context Protocol**：Anthropic 设计的 AI 工具通信协议
- 让 AI 能调用外部工具（文件系统、数据库、网络等）
- RSS MCP 是一个 MCP 服务器，提供 26 个工具

### 3. 通信方式

- **stdio 模式**：通过标准输入输出通信（当前使用）
- **http 模式**：通过 HTTP API 通信（可选）

## 已配置内容

### 订阅源（3 个）

| 订阅源 | 类型 | 更新频率 | 语言 |
|--------|------|---------|------|
| arXiv Quantitative Finance | 学术论文 | 每日 | 英文 |
| BigQuant 博客 | 技术博客 | 不定期 | 中文 |
| Quantpedia | 策略数据库 | 每周 | 英文 |

### 当前文章数

41 篇（截至 2026-04-27）

### 可用工具（26 个）

- **基础**：增删改查订阅、搜索文章
- **分析**：趋势检测、情感分析、去重
- **内容**：翻译、完整内容提取
- **导出**：JSON/CSV/XML、每日摘要
- **AI**：智能推荐、自动分类
- **管理**：书签、定时任务、健康监控

## 使用方式

### 最简单：Shell 脚本

```bash
# 更新订阅
./scripts/rss.sh update

# 搜索文章
./scripts/rss.sh search momentum

# 查看趋势
./scripts/rss.sh trends 7
```

### 编程集成：Python

```python
from scripts.rss_client import RSSMCP

rss = RSSMCP()
articles = rss.search(keyword="momentum", limit=10)
```

### 完整功能：原始命令

查看 `docs/rss_mcp_examples.md`

## 文档索引

| 文档 | 说明 |
|------|------|
| [rss_usage_guide.md](rss_usage_guide.md) | **使用指南**（推荐阅读） |
| [rss_mcp_examples.md](rss_mcp_examples.md) | 完整命令示例 |
| [rss_feeds.md](rss_feeds.md) | 订阅源列表 |

## 快速开始

```bash
# 1. 查看帮助
./scripts/rss.sh help

# 2. 更新订阅
./scripts/rss.sh update

# 3. 搜索最新文章
./scripts/rss.sh search

# 4. 按关键词搜索
./scripts/rss.sh search "factor investing"
```

## 推荐工作流

### 每日

```bash
./scripts/rss.sh update
./scripts/rss.sh search
```

### 每周

```bash
./scripts/rss.sh digest 7
./scripts/rss.sh trends 7
```

### 研究时

```bash
# 搜索特定主题
./scripts/rss.sh search "momentum"

# 找到论文后，用 paper-to-vectorbt 转换为代码
# （在 Claude Code 中使用）
```

## 数据位置

```
/Users/huminghe/.mcp-servers/rss-mcp/
├── data/
│   └── rss.db              # SQLite 数据库（所有数据）
├── src/                    # RSS MCP 源代码
├── node_modules/           # 依赖
└── package.json            # 项目配置

/Users/huminghe/Documents/projects/quant-mh/
├── .mcp.json               # MCP 配置
├── scripts/
│   ├── rss.sh              # Shell 脚本
│   └── rss_client.py       # Python 脚本
└── docs/
    ├── rss_usage_guide.md  # 使用指南
    ├── rss_mcp_examples.md # 命令示例
    └── rss_feeds.md        # 订阅源列表
```

## 常见问题

**Q: RSS MCP 需要一直运行吗？**

A: 不需要。每次调用脚本时会自动启动，执行完自动退出。

**Q: 数据存在哪里？**

A: SQLite 数据库：`/Users/huminghe/.mcp-servers/rss-mcp/data/rss.db`

**Q: 如何添加新订阅？**

A: `./scripts/rss.sh add "RSS_URL" "名称" "分类"`

**Q: 如何备份数据？**

A: `cp /Users/huminghe/.mcp-servers/rss-mcp/data/rss.db ~/backups/`

**Q: Claude Code 能直接调用 RSS MCP 吗？**

A: 理论上可以，但当前通过脚本调用更可靠。

## 下一步

1. **设置自动更新**：添加 crontab 定时任务
2. **扩展订阅源**：添加更多量化相关 RSS
3. **集成工作流**：与 paper-to-vectorbt 结合使用
4. **数据分析**：导出数据进行深度分析

## 技术细节

### MCP 协议格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "rss_search",
    "arguments": {
      "keyword": "momentum",
      "limit": 10
    }
  }
}
```

### 响应格式

```json
{
  "result": {
    "content": [...],
    "structuredContent": {
      "articles": [...]
    }
  },
  "jsonrpc": "2.0",
  "id": 1
}
```

## 维护

### 更新 RSS MCP

```bash
cd /Users/huminghe/.mcp-servers/rss-mcp
git pull
npm install
```

### 清理旧数据

RSS MCP 会自动管理，无需手动清理。如需要：

```bash
# 备份后删除数据库
cp data/rss.db data/rss.db.backup
rm data/rss.db
# 下次运行会自动创建新数据库
```

## 支持

- **RSS MCP 文档**：`/Users/huminghe/.mcp-servers/rss-mcp/README.md`
- **MCP 协议**：https://modelcontextprotocol.io/
- **项目文档**：`docs/` 目录
