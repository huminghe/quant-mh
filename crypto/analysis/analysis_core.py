"""
分析核心引擎
统一调度所有分析模块，支持多标的多策略和单标的多版本两种模式。

用法（由入口脚本调用）：
    from analysis_core import run_all, merge_html

    files = {
        "BTC_v1": "/path/to/btc_v1.xlsx",
        "ETH_v2": "/path/to/eth_v2.xlsx",
        ...
    }
    run_all(files, title="多标的分析", out_base="~/Documents/...")
"""
import datetime
import os
import re
import sys
from pathlib import Path

# 确保同目录模块可 import
sys.path.insert(0, str(Path(__file__).parent))


# ─── HTML 合并 ────────────────────────────────────────────────────────────────

def merge_html(html_files: list[tuple[str, str]], out_path: str, title: str = "综合分析"):
    """将多个 plotly HTML 文件合并为一个带标签页的单文件。"""
    plotly_config = ""
    plotly_bundle = ""
    for _, fp in html_files:
        if not Path(fp).exists():
            continue
        content = Path(fp).read_text(encoding="utf-8")
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        if len(scripts) >= 2:
            plotly_config = f'<script>{scripts[0]}</script>'
            plotly_bundle = f'<script>{max(scripts, key=len)}</script>'
            break

    tabs_html = []
    panels_html = []

    for i, (label, fp) in enumerate(html_files):
        if not Path(fp).exists():
            continue
        content = Path(fp).read_text(encoding="utf-8")
        div_ids = re.findall(r'<div id="([^"]+)"[^>]*class="plotly-graph-div"', content)
        if not div_ids:
            continue

        patched = content
        for old_id in div_ids:
            new_id = f"t{i}_{old_id.replace('-', '')}"
            patched = patched.replace(old_id, new_id)

        divs = re.findall(
            r'<div id="t\d+_[^"]*"[^>]*class="plotly-graph-div"[^>]*>.*?</div>',
            patched, re.DOTALL)

        init_scripts = re.findall(
            r'<script[^>]*>\s*window\.PLOTLYENV.*?</script>', patched, re.DOTALL)

        panel_content = "\n".join(divs) + "\n" + "\n".join(init_scripts)
        active_tab = "active" if i == 0 else ""
        active_panel = "block" if i == 0 else "none"

        tabs_html.append(
            f'<button class="tab-btn {active_tab}" onclick="showTab({i})">{label}</button>')
        panels_html.append(
            f'<div id="panel-{i}" class="tab-panel" style="display:{active_panel}">'
            f'{panel_content}</div>')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
{plotly_config}
{plotly_bundle}
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f5f5; }}
  .tab-bar {{
    display: flex; flex-wrap: wrap; gap: 4px;
    background: #1f3864; padding: 10px 16px;
    position: sticky; top: 0; z-index: 100;
  }}
  .tab-btn {{
    padding: 7px 18px; border: none; border-radius: 4px;
    background: rgba(255,255,255,0.15); color: #fff;
    cursor: pointer; font-size: 13px; transition: background 0.2s;
  }}
  .tab-btn:hover {{ background: rgba(255,255,255,0.3); }}
  .tab-btn.active {{ background: #fff; color: #1f3864; font-weight: bold; }}
  .tab-panel {{ padding: 12px 8px; }}
</style>
</head>
<body>
<div class="tab-bar">
{"".join(tabs_html)}
</div>
{"".join(panels_html)}
<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-panel').forEach((p, i) => {{
    p.style.display = i === idx ? 'block' : 'none';
  }});
  document.querySelectorAll('.tab-btn').forEach((b, i) => {{
    b.classList.toggle('active', i === idx);
  }});
}}
</script>
</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    size_mb = Path(out_path).stat().st_size / 1_000_000
    print(f"  合并 HTML 已保存: {out_path}  ({size_mb:.1f}MB)")


# ─── PNG 导出 ─────────────────────────────────────────────────────────────────

def export_png(html_path: str, png_path: str, width: int = 1200, height: int = 700):
    """
    将 plotly HTML 对应的图表导出为 PNG。
    需要 kaleido 包（pip install kaleido）。
    """
    try:
        import plotly.io as pio
        # 从 HTML 重新生成 figure 不可行，改用 write_image 直接从 figure 对象导出
        # 此函数由各模块在生成 figure 时调用，这里仅作占位
        print(f"  PNG 导出需从 figure 对象调用，跳过: {png_path}")
    except Exception as e:
        print(f"  PNG 导出失败: {e}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_all(
    files: dict[str, str],
    title: str,
    out_base: str,
    window: int = 12,
    date_str: str | None = None,
    llm: bool = False,
) -> str:
    """
    统一分析入口。

    参数：
        files:    {key: 绝对路径}，key 格式为 ASSET_strat（如 BTC_v2、V1_eth）
        title:    分析标题，用于合并 HTML 的标题和文件名前缀
        out_base: 输出根目录
        window:   滚动窗口月数
        date_str: 日期字符串（默认今天）
        llm:      是否在分析完成后自动生成 LLM 解读

    返回：
        out_dir: 输出目录路径
    """
    import analyze_strategies
    import plot_equity_curves
    import plot_combos
    import rolling_analysis
    import heatmap_analysis
    import correlation_matrix
    import generate_detail_report
    import report_md

    if date_str is None:
        date_str = datetime.date.today().isoformat()

    # 创建输出目录
    safe_title = re.sub(r'[^\w\u4e00-\u9fff-]', '_', title)
    out_dir = Path(out_base) / f"charts_{safe_title}_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)

    print(f"\n输出目录: {out_dir_str}\n")
    print("=" * 50)

    # 1. 指标汇总 Excel
    print("【1/7】生成指标汇总 Excel 报告...")
    try:
        analyze_strategies.run(files, base="", out_dir=out_dir_str, date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    # 2. 收益走势图
    print("\n【2/7】生成收益走势图...")
    equity_figs = {}
    try:
        equity_figs = plot_equity_curves.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    # 3. 组合收益回撤图
    print("\n【3/7】生成组合收益回撤图...")
    try:
        plot_combos.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    # 4. 滚动窗口分析
    print("\n【4/7】生成滚动窗口分析...")
    try:
        rolling_analysis.run(files, base="", out_dir=out_dir_str, window=window)
    except Exception as e:
        print(f"  警告: {e}")

    # 5. 热力图 & 相关性矩阵
    print("\n【5/7】生成热力图 & 相关性矩阵...")
    try:
        heatmap_analysis.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")
    try:
        correlation_matrix.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    # 6. 详细分析 Excel
    print("\n【6/7】生成详细分析 Excel 报告...")
    try:
        generate_detail_report.run(files, base="", out_dir=out_dir_str, date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    # 7. 合并 HTML + 导出 PNG + 生成 md 结论
    print("\n【7/7】合并 HTML、导出 PNG、生成结论文档...")

    tab_order = [
        ("收益曲线对比",    "图1_各标的策略对比.html"),
        ("融合策略&全组合", "图2_融合策略与全组合.html"),
        ("关键指标柱状图",  "图3_关键指标柱状对比.html"),
        ("两两组合",        "图4_两两组合收益回撤.html"),
        ("三版本组合",      "图5_三标的全组合收益回撤.html"),
        ("全组合总览",      "图6_所有组合总览.html"),
        ("滚动窗口分析",    "滚动窗口分析.html"),
        ("月度收益热力图",  "热力图_单策略月度收益.html"),
        ("组合热力图",      "热力图_组合月度收益.html"),
        ("相关性矩阵",      "相关性矩阵.html"),
    ]
    available = [(label, str(out_dir / fname))
                 for label, fname in tab_order
                 if (out_dir / fname).exists()]

    # 合并 HTML
    if available:
        combined_path = str(out_dir / f"{title}_{date_str}.html")
        try:
            merge_html(available, combined_path, title=title)
        except Exception as e:
            print(f"  警告: 合并 HTML 失败 {e}")
    else:
        print("  没有可合并的 HTML 文件")

    # 生成 md 结论文档（含 PNG）
    conclusion_path = None
    try:
        report_md.run(files, base="", out_dir=out_dir_str,
                      date_str=date_str, title=title)
        # 找到刚生成的结论文件
        candidates = list(out_dir.glob(f"*_结论_*.md"))
        if candidates:
            conclusion_path = max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"  警告: md 结论文档生成失败 {e}")

    # LLM 解读（可选）
    if llm and conclusion_path:
        print("\n  生成 LLM 解读...")
        try:
            from run_llm_analysis import run_llm_analysis
            run_llm_analysis(conclusion_path)
        except Exception as e:
            print(f"  警告: LLM 解读失败 {e}")

    # 输出文件列表
    print(f"\n全部完成！输出目录: {out_dir_str}")
    print("文件列表:")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        size_str = (f"{size/1024:.0f}{unit}" if size < 1_000_000
                    else f"{size/1_000_000:.1f}{unit}")
        print(f"  {f.name}  ({size_str})")

    return out_dir_str


# ─── 文件扫描工具 ─────────────────────────────────────────────────────────────

DATE_PATTERN = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def scan_files(
    directory: str,
    patterns: list[tuple[str, str]],
    exclude: set[str] | None = None,
) -> dict[str, str]:
    """
    通用文件扫描器。

    参数：
        directory: 扫描目录
        patterns:  [(正则, key), ...]，按顺序匹配文件名（小写），先匹配先得
        exclude:   需要排除的 key 集合

    返回：
        {key: 绝对路径}，同一 key 有多个文件时取日期最新的
    """
    d = Path(directory)
    xlsx_files = list(d.glob("*.xlsx"))
    if not xlsx_files:
        return {}

    # 按日期分组，取最新
    dated: list[tuple[datetime.date, Path]] = []
    for f in xlsx_files:
        m = DATE_PATTERN.search(f.name)
        if m:
            dated.append((datetime.date.fromisoformat(m.group(1)), f))

    if not dated:
        return {}

    latest_date = max(dt for dt, _ in dated)
    latest_files = [f for dt, f in dated if dt == latest_date]

    result: dict[str, str] = {}
    for f in latest_files:
        name = f.name.lower()
        for pattern, key in patterns:
            if re.match(pattern, name):
                if exclude and key in exclude:
                    print(f"  跳过（已排除）: {f.name}")
                    break
                # 同 key 多文件取文件名字典序最大（通常对应最新）
                if key not in result or f.name > Path(result[key]).name:
                    result[key] = str(f)
                break

    return result


# 从文件名自动解析 key 的正则
# 支持格式：
#   strategy_ema_{asset}_OKX_...        → {ASSET}_ema
#   v{N}_strategy_{asset}_OKX_...       → {ASSET}_v{N}
#   v{N}_{tf}_strategy_{asset}_OKX_...  → {ASSET}_v{N}_{tf}
_AUTO_PATTERN = re.compile(
    r"^(?:"
    r"strategy_ema_(?P<asset1>[a-z]+)_"           # strategy_ema_{asset}_
    r"|v(?P<ver>\d+)(?:_(?P<tf>[a-z0-9]+))?_strategy_(?P<asset2>[a-z]+)_"  # v{N}[_{tf}]_strategy_{asset}_
    r")"
)

# 标的别名（文件名中的名称 → 标准名称）
_ASSET_ALIASES = {
    "meme": "DOGE",
}


def scan_files_auto(
    directory: str,
    exclude: set[str] | None = None,
    asset: str | None = None,
) -> dict[str, str]:
    """
    自动扫描并解析文件名，无需手动维护 PATTERNS。

    文件名规则：
      strategy_ema_{asset}_OKX_...        → {ASSET}_ema
      v{N}_strategy_{asset}_OKX_...       → {ASSET}_v{N}
      v{N}_{tf}_strategy_{asset}_OKX_...  → {ASSET}_v{N}_{tf}

    参数：
        directory: 扫描目录
        exclude:   需要排除的 key 集合（如 {"DOGE_ema"}）
        asset:     只保留该标的（如 "ETH"），None 表示全部

    返回：
        {key: 绝对路径}，同一 key 有多个文件时取日期最新的
    """
    d = Path(directory)
    xlsx_files = list(d.glob("*.xlsx"))
    if not xlsx_files:
        return {}

    # 按日期分组，取最新
    dated: list[tuple[datetime.date, Path]] = []
    for f in xlsx_files:
        m = DATE_PATTERN.search(f.name)
        if m:
            dated.append((datetime.date.fromisoformat(m.group(1)), f))

    if not dated:
        return {}

    latest_date = max(dt for dt, _ in dated)
    latest_files = [f for dt, f in dated if dt == latest_date]

    result: dict[str, str] = {}
    for f in latest_files:
        name = f.name.lower()
        m = _AUTO_PATTERN.match(name)
        if not m:
            continue

        # 解析标的名
        raw_asset = m.group("asset1") or m.group("asset2")
        parsed_asset = _ASSET_ALIASES.get(raw_asset, raw_asset).upper()

        # 只保留指定标的
        if asset and parsed_asset != asset.upper():
            continue

        # 解析策略版本
        ver = m.group("ver")
        tf  = m.group("tf")
        if ver is None:
            strat = "ema"
        elif tf:
            strat = f"v{ver}_{tf}"
        else:
            strat = f"v{ver}"

        key = f"{parsed_asset}_{strat}"

        if exclude and key in exclude:
            print(f"  跳过（已排除）: {f.name}")
            continue

        # 同 key 多文件取日期最新的
        if key not in result:
            result[key] = str(f)
        else:
            existing_date = DATE_PATTERN.search(Path(result[key]).name)
            new_date = DATE_PATTERN.search(f.name)
            if new_date and existing_date and new_date.group(1) > existing_date.group(1):
                result[key] = str(f)

    return result
