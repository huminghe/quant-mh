# 扩展 RSS 订阅源 - 问题与解决方案

## 遇到的问题

在尝试添加 Reddit 订阅时遇到超时错误：
```
Network error: timeout of 5000ms exceeded
```

## 原因分析

1. **RSS MCP 超时设置太短**（5 秒）
2. **Reddit 直接访问可能被限制**
3. **RSSHub 公共实例可能不稳定**

## 解决方案

### 方案 1：手动添加（推荐）

等 RSS MCP 更新超时设置后，或者你可以手动修改 RSS MCP 的超时配置。

### 方案 2：使用本地 RSSHub

自己搭建 RSSHub 实例，更稳定：

```bash
# 使用 Docker
docker run -d --name rsshub -p 1200:1200 diygod/rsshub

# 然后使用本地实例
http://localhost:1200/reddit/subreddit/algotrading
```

### 方案 3：使用其他 RSS 阅读器

如果 RSS MCP 不稳定，可以使用专业的 RSS 阅读器：

**Mac 平台：**
- NetNewsWire（免费，开源）
- Reeder（付费，$9.99）
- Feedly（网页版，有免费版）

**跨平台：**
- Inoreader（网页版）
- The Old Reader（网页版）
- FreshRSS（自托管）

### 方案 4：暂时使用现有订阅

当前已成功配置的订阅源：
- ✅ arXiv Quantitative Finance
- ✅ BigQuant 博客
- ✅ Quantpedia

这些已经足够获取高质量的量化研究内容。

## 推荐的订阅源（已验证可用）

### 学术论文
- ✅ arXiv q-fin: http://export.arxiv.org/rss/q-fin
- ⏳ SSRN（需要 VPN）

### 技术博客
- ✅ BigQuant: http://feed.cnblogs.com/blog/u/809308/rss/
- ✅ Quantpedia: https://quantpedia.com/feed/

### 社区讨论（需要 RSSHub 或 VPN）
- ⏳ Reddit r/algotrading
- ⏳ Reddit r/quantfinance
- ⏳ Reddit r/quant

### 国内平台（需要 RSSHub）
- ⏳ 雪球
- ⏳ 知乎专栏

## 当前建议

**短期（现在）：**
- 使用现有的 3 个订阅源
- 每天更新：`./scripts/rss.sh update`
- 搜索感兴趣的主题

**中期（1-2 周后）：**
- 等 RSS MCP 更新或修复超时问题
- 或者自己搭建 RSSHub
- 添加更多订阅源

**长期（1 个月后）：**
- 评估 RSS MCP 的稳定性
- 如果不稳定，考虑迁移到专业 RSS 阅读器
- 或者自己实现简单的 RSS 抓取脚本

## 替代方案：简单的 RSS 抓取脚本

如果 RSS MCP 不够用，可以自己写一个简单的：

```python
import feedparser
import sqlite3
from datetime import datetime

# 订阅源列表
feeds = [
    "http://export.arxiv.org/rss/q-fin",
    "http://feed.cnblogs.com/blog/u/809308/rss/",
    "https://quantpedia.com/feed/",
]

# 抓取并存储
for feed_url in feeds:
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        # 存储到数据库
        save_to_db(entry)
```

## 总结

**当前状态：**
- ✅ 3 个高质量订阅源正常工作
- ⏳ Reddit 等社区订阅遇到网络问题
- ⏳ 等待解决或使用替代方案

**下一步：**
1. 先用现有订阅源
2. 观察 RSS MCP 稳定性
3. 必要时考虑替代方案

**重要提醒：**
- 现有的 3 个订阅源（arXiv、BigQuant、Quantpedia）已经足够获取高质量内容
- Reddit 等社区内容虽然有价值，但不是必需的
- 可以先专注于使用现有资源
