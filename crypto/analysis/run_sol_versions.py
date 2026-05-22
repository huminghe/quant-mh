"""
SOL 多版本策略分析入口
自动扫描 Downloads 目录中所有 SOL 策略文件，无需手动维护版本列表。

用法：
  python run_sol_versions.py
  python run_sol_versions.py --dir ~/Downloads/ --out ~/Documents/.../crypto/analysis/
  python run_sol_versions.py --llm          # 分析完自动生成 LLM 解读
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import run_all, scan_files_auto

TITLE = "SOL多版本策略分析"


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--out", default=os.path.expanduser(
                            "~/Documents/projects/quant-mh/crypto/analysis/"),
                        help="输出根目录")
    parser.add_argument("--window", type=int, default=12,
                        help="滚动窗口月数（默认 12）")
    parser.add_argument("--llm", action="store_true",
                        help="分析完自动生成 LLM 解读")
    args = parser.parse_args()

    print(f"扫描目录: {args.dir}")
    files = scan_files_auto(args.dir, asset="SOL")

    if not files:
        print("未找到任何 SOL 版本文件，请检查 --dir 路径。")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：")
    for k, v in sorted(files.items()):
        print(f"  {k}: {Path(v).name}")

    run_all(files, title=TITLE, out_base=args.out, window=args.window, llm=args.llm)


if __name__ == "__main__":
    main()
