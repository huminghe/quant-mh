#!/bin/bash

# 通过 RSSHub 添加 Reddit 订阅（更稳定）

cd /Users/huminghe/.mcp-servers/rss-mcp

echo "通过 RSSHub 添加 Reddit 订阅..."

# r/algotrading (通过 RSSHub)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://rsshub.app/reddit/subreddit/algotrading","name":"r/algotrading (RSSHub)","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo ""

# r/quantfinance (通过 RSSHub)
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://rsshub.app/reddit/subreddit/quantfinance","name":"r/quantfinance (RSSHub)","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo ""

# r/quant (通过 RSSHub)
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://rsshub.app/reddit/subreddit/quant","name":"r/quant (RSSHub)","category":"reddit"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo ""
echo "Reddit 订阅添加完成（通过 RSSHub）"
