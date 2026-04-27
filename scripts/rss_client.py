#!/usr/bin/env python3
"""
RSS MCP Python 封装
提供更友好的 Python 接口来使用 RSS MCP
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any


class RSSMCP:
    """RSS MCP 客户端"""

    def __init__(self, mcp_path: str = "/Users/huminghe/.mcp-servers/rss-mcp"):
        self.mcp_path = Path(mcp_path)
        self.node_cmd = "node"
        self.tsx_path = "node_modules/tsx/dist/cli.mjs"
        self.index_path = "src/index.ts"

    def _call(self, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用 RSS MCP 工具"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": arguments
            }
        }

        cmd = [
            self.node_cmd,
            str(self.mcp_path / self.tsx_path),
            str(self.mcp_path / self.index_path)
        ]

        env = {"MCP_TRANSPORT": "stdio"}

        result = subprocess.run(
            cmd,
            input=json.dumps(request),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 忽略调试输出
            text=True,
            cwd=str(self.mcp_path),
            env={**subprocess.os.environ, **env}
        )

        # 解析输出，提取 result 部分
        output = result.stdout

        # 尝试找到 JSON 响应
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{') and '"result"' in line:
                try:
                    response = json.loads(line)
                    return response.get('result', {})
                except json.JSONDecodeError:
                    continue

        # 如果没找到，尝试解析整个输出
        try:
            response = json.loads(output)
            return response.get('result', {})
        except json.JSONDecodeError:
            pass

        return {"error": "Failed to parse response", "raw_output": output[:500]}

    def add_feed(self, url: str, name: str, category: str = "general") -> Dict:
        """添加 RSS 订阅"""
        return self._call("rss_add", {
            "url": url,
            "name": name,
            "category": category
        })

    def list_feeds(self) -> List[Dict]:
        """列出所有订阅"""
        result = self._call("rss_list", {})
        return result.get('structuredContent', {}).get('feeds', [])

    def update_feeds(self, feed_id: Optional[str] = None) -> Dict:
        """更新订阅"""
        args = {"feed_id": feed_id} if feed_id else {}
        return self._call("rss_update", args)

    def search(self, keyword: Optional[str] = None, limit: int = 10,
               start_date: Optional[str] = None, end_date: Optional[str] = None,
               category: Optional[str] = None) -> List[Dict]:
        """搜索文章"""
        args = {"limit": limit}
        if keyword:
            args["keyword"] = keyword
        if start_date:
            args["start_date"] = start_date
        if end_date:
            args["end_date"] = end_date
        if category:
            args["category"] = category

        result = self._call("rss_search", args)
        return result.get('structuredContent', {}).get('articles', [])

    def get_trends(self, days: int = 7, top_n: int = 10) -> Dict:
        """获取趋势分析"""
        return self._call("rss_trends", {
            "days": days,
            "top_n": top_n
        })

    def get_analytics(self) -> Dict:
        """获取统计信息"""
        return self._call("rss_analytics", {})

    def daily_digest(self, days: int = 1, format: str = "markdown") -> str:
        """生成每日摘要"""
        result = self._call("rss_daily_digest", {
            "days": days,
            "format": format
        })
        return result.get('structuredContent', {}).get('digest', '')

    def add_bookmark(self, article_id: str) -> Dict:
        """添加书签"""
        return self._call("rss_bookmark", {
            "action": "add",
            "article_id": article_id
        })

    def list_bookmarks(self) -> List[Dict]:
        """列出书签"""
        result = self._call("rss_bookmark", {"action": "list"})
        return result.get('structuredContent', {}).get('bookmarks', [])

    def export(self, format: str = "json", output_path: Optional[str] = None) -> Dict:
        """导出数据"""
        args = {"format": format}
        if output_path:
            args["output_path"] = output_path
        return self._call("rss_export", args)


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='RSS MCP Python 客户端')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # update 命令
    subparsers.add_parser('update', help='更新所有订阅')

    # list 命令
    subparsers.add_parser('list', help='列出所有订阅')

    # search 命令
    search_parser = subparsers.add_parser('search', help='搜索文章')
    search_parser.add_argument('-k', '--keyword', help='关键词')
    search_parser.add_argument('-l', '--limit', type=int, default=10, help='数量限制')

    # trends 命令
    trends_parser = subparsers.add_parser('trends', help='趋势分析')
    trends_parser.add_argument('-d', '--days', type=int, default=7, help='天数')

    # stats 命令
    subparsers.add_parser('stats', help='统计信息')

    # digest 命令
    digest_parser = subparsers.add_parser('digest', help='生成摘要')
    digest_parser.add_argument('-d', '--days', type=int, default=1, help='天数')

    args = parser.parse_args()

    rss = RSSMCP()

    if args.command == 'update':
        result = rss.update_feeds()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == 'list':
        feeds = rss.list_feeds()
        for feed in feeds:
            print(f"📡 {feed['title']}")
            print(f"   URL: {feed['url']}")
            print(f"   添加时间: {feed['added_date']}")
            print()

    elif args.command == 'search':
        articles = rss.search(keyword=args.keyword, limit=args.limit)
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   {article['link']}")
            print(f"   发布时间: {article['pub_date']}")
            print()

    elif args.command == 'trends':
        result = rss.get_trends(days=args.days)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == 'stats':
        result = rss.get_analytics()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == 'digest':
        digest = rss.daily_digest(days=args.days)
        print(digest)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
