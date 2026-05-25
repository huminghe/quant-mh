"""
一键跑全部分析：多标的 + ETH 单标的 + SOL 单标的

用法：
  python run_all.py
  python run_all.py --llm          # 分析完自动生成 LLM 解读
  python run_all.py --health       # 分析完自动生成健康度报告
  python run_all.py --llm --health # 同时生成 LLM 解读和健康度报告
  python run_all.py --dir ~/Downloads/ --out ~/Documents/.../crypto/analysis/
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import run_all, scan_files_auto


ANALYSES = [
    {"title": "多标的策略分析",   "asset": None,  "exclude": {"DOGE_ema"}},
    {"title": "BTC多版本策略分析", "asset": "BTC", "exclude": None},
    {"title": "ETH多版本策略分析", "asset": "ETH", "exclude": None},
    {"title": "SOL多版本策略分析", "asset": "SOL", "exclude": None},
]


def main():
    parser = argparse.ArgumentParser(description="一键跑全部策略分析")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--out", default=os.path.expanduser(
                            "~/Documents/projects/quant-mh/crypto/analysis/"),
                        help="输出根目录")
    parser.add_argument("--window", type=int, default=12,
                        help="滚动窗口月数（默认 12）")
    parser.add_argument("--llm", action="store_true",
                        help="分析完自动生成 LLM 解读")
    parser.add_argument("--health", action="store_true",
                        help="分析完自动生成健康度报告")
    args = parser.parse_args()

    out_dirs = []

    for cfg in ANALYSES:
        print(f"\n{'='*60}")
        print(f"  {cfg['title']}")
        print(f"{'='*60}")

        files = scan_files_auto(args.dir, asset=cfg["asset"], exclude=cfg["exclude"])
        if not files:
            print(f"  未找到文件，跳过")
            continue

        print(f"  找到 {len(files)} 个策略文件")
        out_dir = run_all(files, title=cfg["title"], out_base=args.out,
                          window=args.window, llm=args.llm)
        out_dirs.append((cfg["title"], out_dir))

    print(f"\n{'='*60}")
    print(f"全部分析完成！共 {len(out_dirs)} 个")
    for title, d in out_dirs:
        print(f"  {title}: {d}")

    # 生成健康度报告（基于多标的分析结论）
    if args.health and out_dirs:
        print(f"\n{'='*60}")
        print("  生成健康度报告")
        print(f"{'='*60}")
        health_script = Path(__file__).parent / "health_report.py"
        subprocess.run([sys.executable, str(health_script), "--latest",
                        "--base-dir", args.out], check=False)


if __name__ == "__main__":
    main()
