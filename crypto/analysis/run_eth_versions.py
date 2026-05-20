"""
ETH 四版本策略对比分析
将 V1/V2/V3/V4 当作 4 个独立"标的"，每个版本只有一个策略（eth），
这样图1 会生成 2×2 子图，每格一个版本，便于单独观察。

所有 HTML 图表最终合并为一个 combined.html，通过标签页切换。

用法：
  python run_eth_versions.py
  python run_eth_versions.py --dir ~/Downloads/ --out ~/Downloads/
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import analyze_strategies
import plot_equity_curves
import plot_combos
import rolling_analysis
import heatmap_analysis
import correlation_matrix
import generate_detail_report


# ─── HTML 合并 ────────────────────────────────────────────────────────────────

def merge_html(html_files: list[tuple[str, str]], out_path: str):
    """
    将多个 plotly HTML 文件合并为一个带标签页的单文件。
    html_files: [(tab_label, filepath), ...]

    plotly 输出结构：
      <script>window.PlotlyConfig = ...</script>
      <script>/* plotly.js bundle */</script>
      <div id="UUID" class="plotly-graph-div" ...></div>
      <script>window.PLOTLYENV=...; if (document.getElementById("UUID")) { Plotly.newPlot("UUID", ...) }</script>
    """
    # 从第一个文件提取 plotly.js bundle（最大的那个 script）
    plotly_config = ""
    plotly_bundle = ""
    for _, fp in html_files:
        content = Path(fp).read_text(encoding="utf-8")
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        if len(scripts) >= 2:
            plotly_config = f'<script>{scripts[0]}</script>'
            # 找最大的 script（plotly bundle）
            bundle = max(scripts, key=len)
            plotly_bundle = f'<script>{bundle}</script>'
            break

    tabs_html = []
    panels_html = []

    for i, (label, fp) in enumerate(html_files):
        content = Path(fp).read_text(encoding="utf-8")

        # 找 div id（UUID 格式）
        div_ids = re.findall(r'<div id="([^"]+)"[^>]*class="plotly-graph-div"', content)
        if not div_ids:
            continue

        # 替换所有 UUID 为带 tab 前缀的 id，避免多图表冲突
        patched = content
        for old_id in div_ids:
            new_id = f"t{i}_{old_id.replace('-', '')}"
            patched = patched.replace(old_id, new_id)

        # 提取 div 容器
        divs = re.findall(r'<div id="t{i}_[^"]*"[^>]*class="plotly-graph-div"[^>]*>.*?</div>'.format(i=i),
                          patched, re.DOTALL)
        if not divs:
            # 回退：直接用 id 前缀匹配
            divs = re.findall(r'<div id="t\d+_[^"]*"[^>]*class="plotly-graph-div"[^>]*>.*?</div>',
                              patched, re.DOTALL)

        # 提取图表初始化 script（含 PLOTLYENV 和 Plotly.newPlot）
        init_scripts = re.findall(
            r'<script[^>]*>\s*window\.PLOTLYENV.*?</script>', patched, re.DOTALL)

        panel_content = "\n".join(divs) + "\n" + "\n".join(init_scripts)

        active_tab = "active" if i == 0 else ""
        active_panel = "block" if i == 0 else "none"

        tabs_html.append(
            f'<button class="tab-btn {active_tab}" onclick="showTab({i})">{label}</button>'
        )
        panels_html.append(
            f'<div id="panel-{i}" class="tab-panel" style="display:{active_panel}">'
            f'{panel_content}</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ETH 四策略版本对比分析</title>
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


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETH 四版本策略对比分析")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--out", default=os.path.expanduser(
                            "~/Documents/projects/quant-mh/crypto/analysis/"),
                        help="输出目录（默认 crypto/analysis/）")
    parser.add_argument("--window", type=int, default=12,
                        help="滚动窗口月数（默认 12）")
    args = parser.parse_args()

    src_dir = Path(args.dir)
    out_base = Path(args.out)

    # key 格式：{VERSION}_eth → asset=V1/V2/V3/V4，strat=eth
    # 这样每个版本成为独立"标的"，图1 生成 2×2 子图
    file_map = {
        "V1_eth": "strategy_ema_eth_OKX_ETHUSDT.P_2026-05-20_ad1c2.xlsx",
        "V2_eth": "v2_strategy_eth_OKX_ETHUSDT.P_2026-05-20_1dd9e.xlsx",
        "V3_eth": "v3_3h_strategy_eth_OKX_ETHUSDT.P_2026-05-20_0ac51.xlsx",
        "V4_eth": "v4_205m_strategy_eth_OKX_ETHUSDT.P_2026-05-20_bfc6a.xlsx",
    }

    files = {}
    for key, fname in file_map.items():
        fp = src_dir / fname
        if not fp.exists():
            print(f"文件不存在，跳过: {fp}")
            continue
        files[key] = str(fp)

    if not files:
        print("未找到任何文件，请检查 --dir 路径。")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：")
    for k, v in sorted(files.items()):
        print(f"  {k}: {Path(v).name}")

    date_str = datetime.date.today().isoformat()
    out_dir = out_base / f"charts_eth_versions_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)
    print(f"\n输出目录: {out_dir_str}\n")

    print("=" * 50)
    print("【1/6】生成指标汇总 Excel 报告...")
    try:
        analyze_strategies.run(files, base="", out_dir=out_dir_str, date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【2/6】生成收益走势图...")
    try:
        plot_equity_curves.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【3/6】生成组合收益回撤图...")
    try:
        plot_combos.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【4/6】生成滚动窗口分析...")
    try:
        rolling_analysis.run(files, base="", out_dir=out_dir_str, window=args.window)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【5/6】生成热力图 & 相关性矩阵...")
    try:
        heatmap_analysis.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")
    try:
        correlation_matrix.run(files, base="", out_dir=out_dir_str)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【6/6】生成详细分析 Excel 报告...")
    try:
        generate_detail_report.run(files, base="", out_dir=out_dir_str, date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    # ─── 合并所有 HTML ────────────────────────────────────────────────────────
    print("\n【合并】将所有 HTML 合并为单文件...")
    tab_order = [
        ("收益曲线对比",     "图1_各标的策略对比.html"),
        ("融合策略&全组合",  "图2_融合策略与全组合.html"),
        ("关键指标柱状图",   "图3_关键指标柱状对比.html"),
        ("两两组合",         "图4_两两组合收益回撤.html"),
        ("三版本组合",       "图5_三标的全组合收益回撤.html"),
        ("全组合总览",       "图6_所有组合总览.html"),
        ("滚动窗口分析",     "滚动窗口分析.html"),
        ("月度收益热力图",   "热力图_单策略月度收益.html"),
        ("组合热力图",       "热力图_组合月度收益.html"),
        ("相关性矩阵",       "相关性矩阵.html"),
    ]
    available = [(label, str(out_dir / fname))
                 for label, fname in tab_order
                 if (out_dir / fname).exists()]
    if available:
        combined_path = str(out_dir / f"ETH四版本综合分析_{date_str}.html")
        try:
            merge_html(available, combined_path)
        except Exception as e:
            print(f"  警告: 合并失败 {e}")
    else:
        print("  没有可合并的 HTML 文件")

    print(f"\n全部完成！输出目录: {out_dir_str}")
    print("文件列表:")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        size_str = f"{size/1024:.0f}{unit}" if size < 1_000_000 else f"{size/1_000_000:.1f}{unit}"
        print(f"  {f.name}  ({size_str})")


if __name__ == "__main__":
    main()
