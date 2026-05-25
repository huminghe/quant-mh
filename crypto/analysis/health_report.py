"""
策略健康度 Agent 日报

读取最新的多标的策略分析结论 MD，结合 health_monitor 阈值，
调用 Claude API 生成结构化健康报告。

用法：
  python health_report.py                  # 自动找最新结论文件
  python health_report.py --latest         # 同上
  python health_report.py <结论文件路径>
  python health_report.py --rr 1.45 --wr 32 --dd 1.2   # 直接传入指标
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


# ─── 健康监控阈值（来自 health_monitor.md）────────────────────────────────────

HEALTH_THRESHOLDS = """
## 健康监控阈值（来自 health_monitor.md）

| 指标 | 历史基准 | 预警阈值 | 说明 |
|------|----------|----------|------|
| 滚动3个月盈亏比 | 2.26 | < 1.58（-30%） | 最核心指标 |
| 滚动3个月胜率 | 37.5% | < 30%（-20%） | 辅助指标 |
| 最大回撤 / 历史最大回撤 | 1.0x | > 1.5x | 触发正式审查 |

## 触发规则

| 情况 | 动作 |
|------|------|
| 单月盈亏比低于1.58 | 记录，继续执行，不调整 |
| 连续2个月盈亏比低于1.58 | 触发审查 |
| 最大回撤超历史最大回撤1.5倍 | 暂停策略，重新评估 |

## 归因框架（触发审查时使用）

1. 消息面冲击？把异常大亏损（超过正常止损1.5倍）剔除，重新计算盈亏比。
2. 横盘磨损？看当前市场是否处于低波动横盘状态（ADX < 20，布林带收窄）。
3. 策略本身失效？排除以上两种情况后，盈亏比仍然持续低位，才考虑策略参数审查。

## 心态管理原则

- 胜率37%的策略，连续8次亏损概率2.5%，属于正常范围
- 要判断策略真正失效，需要至少50-100笔交易的滚动样本
- 单次不符合预期不触发调整，需要连续2次

## 历史参照（盈亏比低谷后均回归均值）

| 时期 | 季度盈亏比 | 之后表现 |
|------|-----------|----------|
| 2021Q3 | 1.35 | 2021Q4反弹至2.02 |
| 2022Q1 | 1.40 | 2022Q2反弹至3.53 |
| 2023Q3 | 1.73 | 2023Q4反弹至2.59 |
| 2026Q2 | 1.14 | 待观察 |
"""

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位量化策略风控分析师，专注于趋势跟踪策略的健康度评估。
你的风格：直接给结论，不废话，不重复数据，聚焦判断和行动建议。
输出格式为 Markdown，使用中文。"""

USER_PROMPT_TEMPLATE = """以下是策略健康监控的阈值体系和当前回测数据：

## 健康监控阈值体系
{thresholds}

## 当前策略数据
{strategy_data}

---

请生成一份结构化健康报告，严格按以下格式输出：

## 健康评分

**综合评分：X/5**（1=危险，2=预警，3=正常，4=良好，5=优秀）

**状态：** healthy / degrading / failing 之一

**核心判断：** 一句话说明当前状态的主要原因

---

## 各标的状态

对每个标的给出简短评估（1-2行），格式：
- **标的名**：状态（正常/观察/预警/停止）— 核心原因

---

## 归因分析

基于三步归因框架（消息面冲击 / 横盘磨损 / 策略失效），判断当前表现的主要原因。

---

## 行动建议

**立即行动：** [需要立即做的事，或"无需行动"]

**下次审查关注点：** [下次月度审查时重点看什么]

**不需要做的：** [明确列出不应该做的操作，防止过度干预]

---

注意：
- 评分要保守，宁可低估不要高估
- 如果数据不足以判断，明确说"数据不足"
- 不要建议调整参数，除非明确触发了审查条件
"""


# ─── 核心逻辑 ─────────────────────────────────────────────────────────────────

def find_latest_conclusion(base_dir: str) -> Path | None:
    """找最新的多标的策略分析结论 MD 文件。"""
    # 优先找多标的分析
    candidates = list(Path(base_dir).glob("charts_多标的*/*_结论_*.md"))
    if not candidates:
        # 退而求其次找任意结论文件
        candidates = list(Path(base_dir).glob("charts_*/*_结论_*.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_manual_data(rr: float | None, wr: float | None, dd: float | None) -> str:
    """从命令行参数构建策略数据文本。"""
    lines = ["（手动输入指标）", ""]
    if rr is not None:
        status = "正常" if rr >= 1.58 else ("预警" if rr >= 1.2 else "危险")
        lines.append(f"- 滚动3个月盈亏比：{rr:.2f}（基准2.26，预警线1.58）→ {status}")
    if wr is not None:
        status = "正常" if wr >= 30 else "预警"
        lines.append(f"- 滚动3个月胜率：{wr:.1f}%（基准37.5%，预警线30%）→ {status}")
    if dd is not None:
        status = "正常" if dd <= 1.5 else "触发审查"
        lines.append(f"- 最大回撤/历史最大回撤：{dd:.1f}x（预警线1.5x）→ {status}")
    return "\n".join(lines)


def run_health_report(strategy_data: str, source_desc: str, out_dir: Path) -> Path:
    """调用 Claude API 生成健康报告，写入文件。"""
    print(f"数据来源：{source_desc}")
    print(f"数据长度：{len(strategy_data)} 字符")
    print("调用 Claude API 生成健康报告...")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                thresholds=HEALTH_THRESHOLDS,
                strategy_data=strategy_data,
            )
        }],
    )

    report_text = message.content[0].text
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    md_lines = [
        f"# 策略健康度报告 {date_str}",
        "",
        f"> 由 Claude API 自动生成，生成时间：{time_str}",
        f"> 数据来源：{source_desc}",
        "",
        "---",
        "",
        report_text,
        "",
        "---",
        "",
        "> 本文件由 `health_report.py` 自动生成。",
        "> 触发审查条件：连续2个月盈亏比 < 1.58，或最大回撤超历史最大回撤1.5倍。",
    ]

    out_path = out_dir / f"health_report_{date_str}.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n健康报告已保存：{out_path}")

    usage = message.usage
    print(f"Token 用量：输入 {usage.input_tokens}，输出 {usage.output_tokens}")

    return out_path


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.expanduser("~/Documents/projects/quant-mh/crypto/analysis/")

    parser = argparse.ArgumentParser(description="策略健康度 Agent 日报")
    parser.add_argument("path", nargs="?", help="结论 MD 文件路径（可选）")
    parser.add_argument("--latest", action="store_true", help="自动找最新结论文件")
    parser.add_argument("--base-dir", default=base_dir, help="charts 目录的父目录")
    # 手动输入指标（不依赖结论文件时使用）
    parser.add_argument("--rr",  type=float, help="滚动3个月盈亏比（如 1.45）")
    parser.add_argument("--wr",  type=float, help="滚动3个月胜率，百分比（如 32）")
    parser.add_argument("--dd",  type=float, help="最大回撤/历史最大回撤倍数（如 1.2）")
    args = parser.parse_args()

    # 确定数据来源
    if args.rr is not None or args.wr is not None or args.dd is not None:
        # 手动输入模式
        strategy_data = build_manual_data(args.rr, args.wr, args.dd)
        source_desc = "手动输入指标"
        out_dir = Path(args.base_dir)
    else:
        # 从结论 MD 文件读取
        if args.path:
            conclusion_path = Path(args.path)
            if not conclusion_path.is_absolute():
                conclusion_path = Path(args.base_dir) / conclusion_path
        else:
            # 自动找最新
            conclusion_path = find_latest_conclusion(args.base_dir)
            if not conclusion_path:
                print(f"在 {args.base_dir} 下未找到任何结论 MD 文件")
                print("提示：先运行 run_all.py 生成分析结论，或用 --rr/--wr/--dd 手动输入指标")
                sys.exit(1)
            print(f"自动选择最新结论文件：{conclusion_path}")

        if not conclusion_path.exists():
            print(f"文件不存在：{conclusion_path}")
            sys.exit(1)

        # 读取并清理内容（去掉图片引用）
        content = conclusion_path.read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if not l.strip().startswith("![")]
        strategy_data = "\n".join(lines)
        source_desc = str(conclusion_path.name)
        out_dir = conclusion_path.parent

    run_health_report(strategy_data, source_desc, out_dir)


if __name__ == "__main__":
    main()
