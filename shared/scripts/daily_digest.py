#!/usr/bin/env python3
"""
量化研究每日摘要生成器
用法：python3 daily_digest.py [--days N]
输出：shared/daily_digest/YYYY-MM-DD.md
"""

import json
import subprocess
import sys
import os
from datetime import datetime, date
from email.utils import parsedate_to_datetime

# ── 配置 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSS_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "rss.sh")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "daily_digest")
RSS_MCP_PATH = "/Users/huminghe/.mcp-servers/rss-mcp"


def rss_call(tool: str, args: dict) -> dict:
    """调用 RSS MCP 工具"""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args}
    })
    cmd = (
        f"echo '{payload}' | "
        f"MCP_TRANSPORT=stdio node node_modules/tsx/dist/cli.mjs src/index.ts 2>/dev/null"
    )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=RSS_MCP_PATH
    )
    # MCP 服务器启动日志混在 stdout，找最后一行有效 JSON
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"result"' in line:
            try:
                data = json.loads(line)
                structured = data.get("result", {}).get("structuredContent")
                if structured:
                    return structured
                text = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
                return json.loads(text) if isinstance(text, str) else text
            except Exception:
                continue
    print(f"RSS 调用失败 ({tool}): 无法解析响应", file=sys.stderr)
    return {}


def parse_pub_date(raw: str) -> str:
    """将 RFC 2822 或 ISO 日期字符串统一转为 YYYY-MM-DD，解析失败返回原始前10字符"""
    if not raw:
        return ""
    try:
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # 尝试 ISO 格式
    try:
        return datetime.fromisoformat(raw[:19]).strftime("%Y-%m-%d")
    except Exception:
        return raw[:10]


def fetch_articles(days: int) -> list[dict]:
    """拉取最近 N 天的文章，并注入 feed_name"""
    # 先更新订阅
    print("更新 RSS 订阅...")
    rss_call("rss_update", {})

    # 拉取订阅列表，构建 feed_id → name 映射
    feeds_data = rss_call("rss_list", {})
    feed_map: dict[str, str] = {}
    feeds = []
    if isinstance(feeds_data, dict):
        feeds = feeds_data.get("feeds", [])
    elif isinstance(feeds_data, list):
        feeds = feeds_data
    for f in feeds:
        fid = f.get("id", "")
        name = f.get("title") or f.get("name") or fid
        if fid:
            feed_map[fid] = name

    # 用 search 拉取完整列表（limit 50）
    print(f"拉取最近 {days} 天文章...")
    search = rss_call("rss_search", {"limit": 50})

    articles = []
    if isinstance(search, list):
        articles = search
    elif isinstance(search, dict):
        articles = search.get("articles", search.get("items", []))

    # 注入 feed_name
    for a in articles:
        if not a.get("feed_name"):
            fid = a.get("feed_id", "")
            a["feed_name"] = feed_map.get(fid, "其他")

    return articles


def summarize_with_claude(title: str, abstract: str) -> str:
    """用 claude -p 生成一句话中文摘要"""
    if not abstract or len(abstract.strip()) < 20:
        return "（无摘要）"

    prompt = f"""请用一句话（30字以内）概括以下量化金融文章的核心内容，用中文回答，不要加任何前缀：

标题：{title}
摘要：{abstract[:500]}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        summary = result.stdout.strip()
        # 去掉可能的引号或多余格式
        summary = summary.strip('"').strip("'").strip()
        return summary if summary else "（摘要生成失败）"
    except subprocess.TimeoutExpired:
        return "（摘要超时）"
    except Exception as e:
        return f"（{str(e)[:20]}）"


def group_by_source(articles: list[dict]) -> dict[str, list]:
    """按来源分组"""
    groups: dict[str, list] = {}
    for a in articles:
        source = a.get("feed_name") or a.get("source") or a.get("feed") or "其他"
        groups.setdefault(source, []).append(a)
    return groups


def generate_markdown(articles: list[dict], target_date: date) -> str:
    """生成 Markdown 内容"""
    date_str = target_date.strftime("%Y-%m-%d")
    lines = [
        f"# 量化研究日报 {date_str}",
        "",
        f"> 共 {len(articles)} 篇文章，来自 {len(set(a.get('feed_name','其他') for a in articles))} 个订阅源",
        "",
    ]

    if not articles:
        lines.append("今日暂无新文章。")
        return "\n".join(lines)

    groups = group_by_source(articles)

    for source, items in groups.items():
        lines.append(f"## {source}（{len(items)} 篇）")
        lines.append("")
        for i, article in enumerate(items, 1):
            title = article.get("title", "无标题").strip()
            url = article.get("url") or article.get("link") or ""
            abstract = article.get("summary") or article.get("description") or article.get("content") or ""
            pub_date = article.get("published") or article.get("pub_date") or ""

            print(f"  [{i}/{len(items)}] 生成摘要：{title[:40]}...")
            summary = summarize_with_claude(title, abstract)

            # 标题行
            if url:
                lines.append(f"- **[{title}]({url})**")
            else:
                lines.append(f"- **{title}**")

            # 摘要和日期
            lines.append(f"  {summary}")
            if pub_date:
                lines.append(f"  *{parse_pub_date(pub_date)}*")
            lines.append("")

        lines.append("")

    lines += [
        "---",
        "",
        f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成量化研究每日摘要")
    parser.add_argument("--days", type=int, default=1, help="拉取最近 N 天的文章（默认 1）")
    parser.add_argument("--date", type=str, default=None, help="输出文件日期（默认今天，格式 YYYY-MM-DD）")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    output_path = os.path.join(OUTPUT_DIR, f"{target_date}.md")

    print(f"目标日期：{target_date}")
    print(f"输出文件：{output_path}")

    articles = fetch_articles(args.days)
    if not articles:
        print("未获取到文章，请检查 RSS 订阅状态")
        sys.exit(1)

    print(f"\n共 {len(articles)} 篇文章，开始生成摘要...")
    content = generate_markdown(articles, target_date)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n完成！文件已保存到：{output_path}")


if __name__ == "__main__":
    main()
