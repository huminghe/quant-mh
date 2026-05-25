"""
回测结果分析 Agent

读取回测脚本导出的 JSON 结果，调用 Claude API 生成：
1. 结构化分析报告（归因、风险、改进方向）
2. 下一步研究假设（可直接执行的方向）

用法：
  python backtest_analyst.py reversal_20d_hs300_results.json
  python backtest_analyst.py reversal_20d_hs300_results.json --out ./reports/
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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

SYSTEM_PROMPT = """你是一位量化策略研究员，专注于 A 股多因子策略的研究与迭代。
你的风格：直接给结论，不废话，聚焦判断和可执行的下一步。
输出格式为 Markdown，使用中文。"""

USER_PROMPT_TEMPLATE = """以下是一次回测的完整结果：

```json
{results_json}
```

请生成一份结构化分析报告，严格按以下格式输出：

## 总体评估

**评级：** 优秀 / 良好 / 一般 / 较差（选一个）

**核心判断：** 一句话说明策略当前状态和主要问题

---

## 归因分析

从以下三个维度分析结果：

**1. 因子有效性**
- IC 均值和 IC_IR 是否达标？因子信号质量如何？

**2. 执行层面**
- 交易成本、换手率、持仓集中度是否合理？

**3. 市场环境**
- 当前参数设置是否适配 A 股市场特征（T+1、涨跌停、流动性）？

---

## 风险点

列出 2-3 个最值得关注的风险，每条一行，格式：
- **风险名称**：具体说明

---

## 下一步研究假设

基于当前结果，给出 3 个可直接验证的研究方向，按优先级排序：

**假设 1（最高优先级）：**
- 方向：[具体改进方向]
- 预期效果：[改进后预期指标变化]
- 验证方法：[如何用代码验证，越具体越好]

**假设 2：**
- 方向：
- 预期效果：
- 验证方法：

**假设 3：**
- 方向：
- 预期效果：
- 验证方法：

---

注意：
- 假设必须基于当前数据，不要凭空发散
- 验证方法要具体到参数修改或代码逻辑
- 如果某项指标数据不足以判断，明确说明
"""


# ─── 核心逻辑 ─────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    """加载回测结果 JSON。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_results_summary(results: dict) -> str:
    """把 JSON 结果格式化为可读摘要（同时保留原始 JSON 供 LLM 使用）。"""
    return json.dumps(results, ensure_ascii=False, indent=2)


def run_analysis(results: dict, source_path: Path, out_dir: Path) -> Path:
    """调用 Claude API 生成分析报告，写入文件。"""
    results_json = format_results_summary(results)
    strategy_name = results.get("strategy", {}).get("name", source_path.stem)

    print(f"策略：{strategy_name}")
    print(f"数据长度：{len(results_json)} 字符")
    print("调用 Claude API 生成分析报告...")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(results_json=results_json),
        }],
    )

    report_text = message.content[0].text
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 组装完整报告
    perf = results.get("performance", {})
    bench = results.get("benchmark", {})
    factor = results.get("factor", {})

    header_lines = [
        f"# 回测分析报告：{strategy_name}",
        "",
        f"> 生成时间：{time_str}  |  数据来源：{source_path.name}",
        "",
        "---",
        "",
        "## 关键指标速览",
        "",
        "| 指标 | 策略 | 基准（沪深300） |",
        "|------|------|----------------|",
        f"| 年化收益 | {perf.get('annual_return_pct', 'N/A')}% | {bench.get('annual_return_pct', 'N/A')}% |",
        f"| 夏普比率 | {perf.get('sharpe_ratio', 'N/A')} | — |",
        f"| 最大回撤 | {perf.get('max_drawdown_pct', 'N/A')}% | — |",
        f"| 胜率 | {perf.get('win_rate_pct', 'N/A')}% | — |",
        f"| IC 均值 | {factor.get('ic_mean', 'N/A')} | — |",
        f"| IC_IR | {factor.get('ic_ir', 'N/A')} | — |",
        f"| 超额收益 | {bench.get('excess_return_pct', 'N/A')}% | — |",
        "",
        "---",
        "",
        report_text,
        "",
        "---",
        "",
        "> 本文件由 `backtest_analyst.py` 自动生成。",
    ]

    out_path = out_dir / f"analysis_{source_path.stem}_{date_str}.md"
    out_path.write_text("\n".join(header_lines), encoding="utf-8")
    print(f"\n分析报告已保存：{out_path}")

    usage = message.usage
    print(f"Token 用量：输入 {usage.input_tokens}，输出 {usage.output_tokens}")

    return out_path


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="回测结果分析 Agent")
    parser.add_argument("path", help="回测结果 JSON 文件路径")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认与 JSON 文件同目录）")
    args = parser.parse_args()

    source_path = Path(args.path)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.exists():
        print(f"文件不存在：{source_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(source_path)
    run_analysis(results, source_path, out_dir)


if __name__ == "__main__":
    main()
