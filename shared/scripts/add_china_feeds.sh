#!/bin/bash

# 添加国内可访问的量化相关订阅源

cd /Users/huminghe/.mcp-servers/rss-mcp

echo "添加国内量化订阅源..."

# 雪球用户（通过 RSSHub）
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://rsshub.app/xueqiu/user/1247347556","name":"雪球-量化投资","category":"xueqiu"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo ""

# 知乎专栏（通过 RSSHub）
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rss_add","arguments":{"url":"https://rsshub.app/zhihu/zhuanlan/quanttech","name":"知乎-量化技术","category":"zhihu"}}}' | \
MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 20 '"result"'

echo ""
echo "国内订阅源添加完成"
