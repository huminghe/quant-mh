# RSS MCP 使用指南

## 快速开始

RSS MCP 已配置完成，提供了 3 种使用方式：

### 1. Shell 脚本（推荐）

最简单直接的方式：

```bash
# 更新所有订阅
./scripts/rss.sh update

# 列出订阅
./scripts/rss.sh list

# 搜索最新文章
./scripts/rss.sh search

# 按关键词搜索
./scripts/rss.sh search momentum

# 趋势分析（最近 7 天）
./scripts/rss.sh trends

# 趋势分析（最近 30 天）
./scripts/rss.sh trends 30

# 统计信息
./scripts/rss.sh stats

# 生成每日摘要
./scripts/rss.sh digest

# 生成每周摘要
./scripts/rss.sh digest 7

# 添加新订阅
./scripts/rss.sh add "https://example.com/feed" "订阅名称" "分类"
```

### 2. Python 库（编程集成）

在 Python 代码中使用（命令行工具暂不可用，仅作为库）：

```python
from scripts.rss_client import RSSMCP

rss = RSSMCP()

# 搜索文章
articles = rss.search(keyword="momentum", limit=10)
for article in articles:
    print(f"{article['title']}: {article['link']}")

# 获取趋势
trends = rss.get_trends(days=7)
print(trends)

# 生成摘要
digest = rss.daily_digest(days=1)
print(digest)
```

### 3. 原始命令（完整功能）

查看 `docs/rss_mcp_examples.md` 获取所有 26 个工具的详细用法。

## 当前订阅源

1. **arXiv Quantitative Finance**
   - 学术论文，每日更新
   - 涵盖量化金融最新研究

2. **BigQuant 博客**
   - 中文技术博客
   - Python 实现、策略研究

3. **Quantpedia**
   - 策略数据库，每周更新
   - 回测分析、因子投资

## 推荐工作流

### 每日工作流

```bash
# 1. 早上更新订阅
./scripts/rss.sh update

# 2. 查看最新文章
./scripts/rss.sh search

# 3. 按兴趣搜索
./scripts/rss.sh search "factor"
./scripts/rss.sh search "machine learning"
./scripts/rss.sh search "reversal"
```

### 每周工作流

```bash
# 1. 生成每周摘要
./scripts/rss.sh digest 7

# 2. 查看趋势
./scripts/rss.sh trends 7

# 3. 查看统计
./scripts/rss.sh stats
```

### 研究工作流

当你在研究特定主题时：

```bash
# 1. 搜索相关文章
./scripts/rss.sh search "momentum"

# 2. 找到感兴趣的文章后，可以用 paper-to-vectorbt skill 转换为回测代码
# （在 Claude Code 中使用）
```

## 自动化建议

### 设置每日自动更新

添加到 crontab：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天早上 9 点更新）
0 9 * * * cd /Users/huminghe/Documents/projects/quant-mh && ./scripts/rss.sh update >> logs/rss_update.log 2>&1
```

### 设置每周摘要邮件

创建脚本 `scripts/weekly_digest.sh`：

```bash
#!/bin/bash
cd /Users/huminghe/Documents/projects/quant-mh
./scripts/rss.sh digest 7 > /tmp/rss_digest.txt
# 这里可以添加发送邮件的命令
```

## 高级功能

### 趋势分析

查看最近热门话题：

```bash
./scripts/rss.sh trends 30
```

### 书签管理

保存感兴趣的文章：

```bash
# 列出书签
./scripts/rss.sh bookmark list

# 添加书签（需要文章 ID）
./scripts/rss.sh bookmark add article_xxx
```

### 导出数据

如果需要导出所有数据进行分析：

```bash
cd /Users/huminghe/.mcp-servers/rss-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_export","arguments":{"format":"json","output_path":"./export.json"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

## 数据位置

- **数据库**：`/Users/huminghe/.mcp-servers/rss-mcp/data/rss.db`
- **配置**：`/Users/huminghe/Documents/projects/quant-mh/.mcp.json`
- **脚本**：`/Users/huminghe/Documents/projects/quant-mh/scripts/`

## 故障排查

### 更新失败

```bash
# 检查网络连接
curl -I http://export.arxiv.org/rss/q-fin

# 查看详细错误
cd /Users/huminghe/.mcp-servers/rss-mcp
./scripts/rss.sh update 2>&1 | less
```

### 搜索无结果

```bash
# 确认数据库中有文章
./scripts/rss.sh stats

# 如果没有，先更新
./scripts/rss.sh update
```

## 扩展订阅源

添加新的 RSS 源：

```bash
./scripts/rss.sh add "RSS_URL" "订阅名称" "分类"
```

推荐的量化相关 RSS 源（待验证）：
- 券商研报（需要找到 RSS 源）
- 聚宽社区（需要确认是否有 RSS）
- 知乎量化话题（可能需要第三方 RSS 服务）

## 与 paper-to-vectorbt 集成

当你从 RSS 中发现感兴趣的论文时：

1. 复制论文链接
2. 在 Claude Code 中使用 paper-to-vectorbt skill
3. 自动生成回测代码

示例：
```
/paper-to-vectorbt https://arxiv.org/abs/2602.01022
```

## 性能优化

如果文章数量很大，搜索变慢：

```bash
# 限制搜索结果数量
./scripts/rss.sh search momentum | head -50

# 或使用 Python 脚本的 limit 参数
./scripts/rss_client.py search -k momentum -l 5
```

## 备份与恢复

### 备份数据库

```bash
cp /Users/huminghe/.mcp-servers/rss-mcp/data/rss.db ~/backups/rss_$(date +%Y%m%d).db
```

### 导出订阅列表（OPML）

```bash
cd /Users/huminghe/.mcp-servers/rss-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_opml","arguments":{"action":"export","output_path":"./feeds.opml"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

## 常见问题

**Q: 为什么 Claude Code 没有自动加载 RSS MCP 工具？**

A: 项目级 `.mcp.json` 可能不被识别。可以配置到全局配置文件，但使用脚本已经足够方便。

**Q: 如何删除不需要的订阅？**

A: 先用 `./scripts/rss.sh list` 获取订阅 ID，然后查看 `docs/rss_mcp_examples.md` 中的删除命令。

**Q: 数据库会不会越来越大？**

A: RSS MCP 会自动管理旧文章。如果需要，可以定期备份并清理数据库。

**Q: 可以添加多少个订阅源？**

A: 理论上无限制，但建议控制在 20 个以内，保证更新速度。
