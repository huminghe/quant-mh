#!/bin/bash

# RSS MCP 便捷工具脚本
# 使用方法：./rss.sh <命令> [参数]

RSS_MCP_PATH="/Users/huminghe/.mcp-servers/rss-mcp"

# 执行 RSS 命令的通用函数
rss_call() {
    local tool=$1
    local args=$2
    cd "$RSS_MCP_PATH"
    echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args}}" | \
    MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>&1 | grep -A 500 '"result"' | python3 -m json.tool 2>/dev/null || grep -A 500 '"result"'
}

# 命令处理
case "$1" in
    update)
        echo "📡 更新所有订阅..."
        rss_call "rss_update" "{}"
        ;;

    list)
        echo "📋 订阅列表："
        rss_call "rss_list" "{}"
        ;;

    search)
        if [ -z "$2" ]; then
            echo "🔍 最新 10 篇文章："
            rss_call "rss_search" "{\"limit\":10}"
        else
            echo "🔍 搜索关键词: $2"
            rss_call "rss_search" "{\"keyword\":\"$2\",\"limit\":10}"
        fi
        ;;

    trends)
        days=${2:-7}
        echo "📈 最近 $days 天的趋势分析："
        rss_call "rss_trends" "{\"days\":$days,\"top_n\":10}"
        ;;

    stats)
        echo "📊 统计信息："
        rss_call "rss_analytics" "{}"
        ;;

    digest)
        days=${2:-1}
        echo "📰 最近 $days 天的摘要："
        rss_call "rss_daily_digest" "{\"format\":\"markdown\",\"days\":$days}"
        ;;

    digest-ai)
        days=${2:-1}
        echo "🤖 生成 AI 摘要日报（最近 $days 天）..."
        python3 "$(dirname "$0")/daily_digest.py" --days "$days"
        ;;

    add)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "用法: $0 add <URL> <名称> [分类]"
            exit 1
        fi
        category=${4:-general}
        echo "➕ 添加订阅: $3"
        rss_call "rss_add" "{\"url\":\"$2\",\"name\":\"$3\",\"category\":\"$category\"}"
        ;;

    bookmark)
        if [ "$2" = "list" ]; then
            echo "🔖 书签列表："
            rss_call "rss_bookmark" "{\"action\":\"list\"}"
        elif [ "$2" = "add" ] && [ -n "$3" ]; then
            echo "🔖 添加书签: $3"
            rss_call "rss_bookmark" "{\"action\":\"add\",\"article_id\":\"$3\"}"
        else
            echo "用法: $0 bookmark list|add <article_id>"
            exit 1
        fi
        ;;

    help|--help|-h)
        cat <<EOF
RSS MCP 便捷工具

用法: $0 <命令> [参数]

命令：
  update              更新所有订阅
  list                列出所有订阅
  search [关键词]      搜索文章（不带关键词则显示最新 10 篇）
  trends [天数]        趋势分析（默认 7 天）
  stats               统计信息
  digest [天数]        生成摘要（默认 1 天）
  digest-ai [天数]     生成 AI 中文摘要日报，输出到 shared/daily_digest/（默认 1 天）
  add <URL> <名称> [分类]  添加新订阅
  bookmark list       列出书签
  bookmark add <ID>   添加书签
  help                显示此帮助

示例：
  $0 update
  $0 search momentum
  $0 trends 30
  $0 digest 7
  $0 add "https://example.com/feed" "示例订阅" "blog"

EOF
        ;;

    *)
        echo "未知命令: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac
