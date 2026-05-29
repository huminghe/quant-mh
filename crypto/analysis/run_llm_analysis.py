"""
LLM 策略解读脚本

读取已生成的 _结论_ MD 文件，调用 Claude API 进行深度解读，
输出 _解读_ MD 文件。

用法：
  python run_llm_analysis.py <结论文件路径>
  python run_llm_analysis.py charts_多标的策略分析_2026-05-22/多标的策略分析_结论_2026-05-22.md
  python run_llm_analysis.py --latest  # 自动找最新的结论文件
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 适配非标准环境变量名
if not os.environ.get("ANTHROPIC_API_KEY"):
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if token:
        os.environ["ANTHROPIC_API_KEY"] = token

try:
    import anthropic
except ImportError:
    print("缺少 anthropic 包，请运行：pip install anthropic")
    sys.exit(1)


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的量化策略分析师，擅长趋势跟踪策略的评估与组合优化。
你的分析风格：简洁、直接、有结论，不废话，不重复数据，聚焦洞察和建议。
输出格式为 Markdown，使用中文。"""

USER_PROMPT_TEMPLATE = """以下是一份量化策略回测分析报告的数据摘要：

---
{conclusion_content}
---

请基于以上数据，给出专业的策略解读，包含以下几个方面：

1. **整体评估**：这批策略的整体质量如何？有哪些突出的优点和明显的弱点？

2. **策略分化分析**：各策略/版本之间的表现差异说明了什么？哪些策略值得重点关注，哪些可以考虑降权或放弃？

3. **组合建议**：基于相关性和回撤数据，推荐 1-2 个最优组合方案，说明理由。如果是单标的多版本分析，重点分析版本间的互补性。

4. **风险提示**：当前数据中有哪些值得警惕的风险信号？（如高相关性、特定时期的集中回撤、夏普/Sortino 差距过大等）

5. **下一步建议**：基于当前分析，建议优先做什么？（如：某个版本值得进一步优化、某个组合可以考虑实盘验证、某个标的需要更多数据等）

注意：
- 直接给出结论，不要重复表格里的数字
- 如果数据不足以支撑某个判断，明确说明
- 保持客观，不要过度乐观
"""


# ─── 核心逻辑 ─────────────────────────────────────────────────────────────────

def find_latest_conclusion(base_dir: str) -> Path | None:
    """在 base_dir 下找最新的 _结论_ MD 文件。"""
    candidates = list(Path(base_dir).glob("charts_*/*_结论_*.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_llm_analysis(conclusion_path: Path) -> Path:
    """
    读取结论文件，调用 LLM，写出解读文件。
    返回解读文件路径。
    """
    content = conclusion_path.read_text(encoding="utf-8")

    # 去掉 PNG 图片引用行（LLM 看不到图片）
    lines = [l for l in content.splitlines() if not l.strip().startswith("![")]
    clean_content = "\n".join(lines)

    print(f"读取结论文件: {conclusion_path}")
    print(f"内容长度: {len(clean_content)} 字符")
    print("调用 Claude API...")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                conclusion_content=clean_content
            )}
        ],
    )

    analysis_text = message.content[0].text

    # 组装解读文件
    title = conclusion_path.stem.replace("_结论_", "_解读_")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    md_lines = [
        f"# {title}",
        "",
        f"> 由 Claude API 自动生成，生成时间：{date_str}",
        f"> 数据来源：[{conclusion_path.name}]({conclusion_path.name})",
        "",
        "---",
        "",
        analysis_text,
        "",
        "---",
        "",
        "> 本文件由 `run_llm_analysis.py` 自动生成，可手动补充修改。",
        "> 重新生成不会覆盖本文件（文件名含时间戳）。",
    ]

    out_path = conclusion_path.parent / f"{title}_{datetime.now().strftime('%H%M')}.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"解读文件已保存: {out_path}")

    # 打印 token 用量
    usage = message.usage
    print(f"Token 用量: 输入 {usage.input_tokens}，输出 {usage.output_tokens}")

    return out_path


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM 策略解读")
    parser.add_argument("path", nargs="?", help="结论 MD 文件路径")
    parser.add_argument("--latest", action="store_true",
                        help="自动找最新的结论文件")
    parser.add_argument("--base-dir", default=os.path.expanduser(
                            "~/Documents/projects/quant-mh/crypto/analysis/reports/"),
                        help="charts 目录的父目录")
    args = parser.parse_args()

    if args.latest:
        path = find_latest_conclusion(args.base_dir)
        if not path:
            print(f"在 {args.base_dir} 下未找到任何 _结论_ MD 文件")
            sys.exit(1)
        print(f"自动选择最新结论文件: {path}")
    elif args.path:
        path = Path(args.path)
        if not path.is_absolute():
            # 相对路径先尝试当前目录，再尝试 base_dir
            if (Path.cwd() / path).exists():
                path = Path.cwd() / path
            else:
                path = Path(args.base_dir) / path
        if not path.exists():
            print(f"文件不存在: {path}")
            sys.exit(1)
    else:
        # 无参数时列出可用文件让用户选
        base = Path(args.base_dir)
        candidates = sorted(base.glob("charts_*/*_结论_*.md"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("未找到任何结论文件，请先运行分析脚本生成结论。")
            sys.exit(1)
        print("可用的结论文件：")
        for i, p in enumerate(candidates[:10]):
            print(f"  [{i}] {p.relative_to(base)}")
        choice = input("请输入编号（默认 0）: ").strip() or "0"
        path = candidates[int(choice)]

    run_llm_analysis(path)


if __name__ == "__main__":
    main()
