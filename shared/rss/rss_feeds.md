# 量化研究 RSS 订阅源

> **快速开始**：使用 `./scripts/rss.sh help` 查看所有命令
> 
> **完整指南**：查看 [RSS 使用指南](rss_usage_guide.md)
> 
> **详细示例**：查看 [RSS MCP 示例](rss_mcp_examples.md)

## 已验证可用的订阅源

### 1. arXiv Quantitative Finance
- **URL**: http://export.arxiv.org/rss/q-fin
- **类型**: 学术论文
- **更新频率**: 每日
- **语言**: 英文
- **说明**: 量化金融领域最新学术论文，涵盖投资组合管理、风险管理、交易策略等

### 2. BigQuant 博客
- **URL**: http://feed.cnblogs.com/blog/u/809308/rss/
- **类型**: 技术博客
- **更新频率**: 不定期
- **语言**: 中文
- **说明**: BigQuant 平台的量化策略研究、机器学习应用、Python 实现

### 3. Quantpedia
- **URL**: https://quantpedia.com/feed/
- **类型**: 策略数据库
- **更新频率**: 每周
- **语言**: 英文
- **说明**: 量化策略研究、回测分析、因子投资

## 待验证的订阅源

### SSRN Quantitative Finance
- **URL**: https://papers.ssrn.com/sol3/rss_feed.cfm?journal=1079991
- **状态**: 访问受限（403）
- **备注**: 可能需要 VPN 或账户登录

### Alpha Architect
- **URL**: https://alphaarchitect.com/feed/
- **状态**: 访问受限（403）
- **备注**: 可能需要 VPN

## RSS MCP 配置状态

✅ **已完成配置**（2026-04-27）

- MCP 服务器路径：`/Users/huminghe/.mcp-servers/rss-mcp`
- 配置文件：`/Users/huminghe/Documents/projects/quant-mh/.mcp.json`
- 已添加订阅：3 个
- 当前文章数：41 篇

### 已添加的订阅

1. ✅ arXiv Quantitative Finance
2. ✅ BigQuant 博客
3. ✅ Quantpedia

### 常用命令

```bash
# 进入 RSS MCP 目录
cd /Users/huminghe/.mcp-servers/rss-mcp

# 列出所有订阅
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_list","arguments":{}}}' | MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'

# 更新所有订阅
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_update","arguments":{}}}' | MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 30 '"result"'

# 搜索最新文章（限制 10 篇）
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"limit":10}}}' | MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 200 '"result"'

# 按关键词搜索
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"keyword":"momentum","limit":5}}}' | MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

## 使用建议

- **每日检查**: arXiv（学术前沿）
- **每周检查**: Quantpedia（策略灵感）
- **按需检查**: BigQuant（中文实战案例）

## 后续扩展

可以考虑添加：
- 券商金工研报（需要找到 RSS 源）
- 聚宽社区（需要确认是否有 RSS）
- 知乎量化话题（可能需要第三方 RSS 服务）
