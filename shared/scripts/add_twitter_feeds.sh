#!/bin/bash

# 添加量化相关 Twitter 账号（通过 Nitter）

cd /Users/huminghe/.mcp-servers/rss-mcp

# QuantStreet
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://nitter.net/QuantStreet/rss","name":"QuantStreet Twitter","category":"twitter"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

# Quantocracy
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://nitter.net/quantocracy/rss","name":"Quantocracy Twitter","category":"twitter"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo "Twitter 订阅添加完成"
