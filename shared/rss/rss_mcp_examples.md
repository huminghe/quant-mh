# RSS MCP 使用示例

## 基础命令模板

所有命令都在 RSS MCP 目录下执行：
```bash
cd /Users/huminghe/.mcp-servers/rss-mcp
```

## 1. 订阅管理

### 添加订阅
```bash
cat > request.json <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"RSS_URL","name":"订阅名称","category":"分类"}}}
EOF
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts < request.json 2>&1 | grep -A 20 '"result"'
```

### 列出所有订阅
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_list","arguments":{}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

### 删除订阅
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_delete","arguments":{"feed_id":"订阅ID"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'
```

## 2. 内容获取

### 更新所有订阅
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_update","arguments":{}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 30 '"result"'
```

### 更新特定订阅
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_update","arguments":{"feed_id":"订阅ID"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 30 '"result"'
```

### 获取特定订阅的文章
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_news","arguments":{"feed_id":"订阅ID","limit":10}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

## 3. 搜索与过滤

### 搜索最新文章
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"limit":10}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 200 '"result"'
```

### 按关键词搜索
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"keyword":"momentum","limit":5}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

### 按日期范围搜索
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"start_date":"2026-04-01","end_date":"2026-04-27","limit":20}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 200 '"result"'
```

### 按分类搜索
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_search","arguments":{"category":"academic","limit":10}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 200 '"result"'
```

## 4. 高级分析

### 趋势分析
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_trends","arguments":{"days":7,"top_n":10}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

### 情感分析
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_sentiment_analysis","arguments":{"article_id":"文章ID"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

### 统计分析
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_analytics","arguments":{}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

### 查找重复文章
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_duplicates","arguments":{"threshold":0.8}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

## 5. 导出与报告

### 生成每日摘要
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_daily_digest","arguments":{"format":"markdown","days":1}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 500 '"result"'
```

### 导出为 JSON
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_export","arguments":{"format":"json","output_path":"./export.json"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

### 导出 OPML（订阅列表）
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_opml","arguments":{"action":"export","output_path":"./feeds.opml"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

## 6. 书签管理

### 添加书签
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_bookmark","arguments":{"action":"add","article_id":"文章ID"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'
```

### 列出书签
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_bookmark","arguments":{"action":"list"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 100 '"result"'
```

## 7. 定时任务

### 设置自动更新（每天早上 9 点）
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_schedule","arguments":{"action":"add","cron":"0 9 * * *","task":"update_all"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'
```

### 列出定时任务
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_schedule","arguments":{"action":"list"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 50 '"result"'
```

## 快捷脚本

为了方便使用，可以创建 shell 函数：

```bash
# 添加到 ~/.zshrc 或 ~/.bashrc

# RSS MCP 基础路径
RSS_MCP_PATH="/Users/huminghe/.mcp-servers/rss-mcp"

# 执行 RSS 命令
rss_call() {
    local tool=$1
    local args=$2
    cd "$RSS_MCP_PATH"
    echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args}}" | \
    MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 200 '"result"'
}

# 快捷命令
alias rss-update='rss_call "rss_update" "{}"'
alias rss-list='rss_call "rss_list" "{}"'
alias rss-search='rss_call "rss_search" "{\"limit\":10}"'
alias rss-stats='rss_call "rss_analytics" "{}"'
```

使用示例：
```bash
source ~/.zshrc
rss-update      # 更新所有订阅
rss-list        # 列出订阅
rss-search      # 搜索最新文章
rss-stats       # 查看统计
```
