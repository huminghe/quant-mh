#!/bin/bash

# 添加量化相关 Reddit 订阅

cd /Users/huminghe/.mcp-servers/rss-mcp

# r/algotrading
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://www.reddit.com/r/algotrading/.rss","name":"r/algotrading","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

# r/quantfinance
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://www.reddit.com/r/quantfinance/.rss","name":"r/quantfinance","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

# r/quant
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://www.reddit.com/r/quant/.rss","name":"r/quant","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo "Reddit 订阅添加完成"
