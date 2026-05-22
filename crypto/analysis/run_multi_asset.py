"""
多标的多策略分析入口
自动扫描 Downloads 目录，解析文件名推导标的和策略版本，无需手动维护匹配规则。

支持任意版本命名，如 v7、v9_321m 等，新增策略无需改代码。

用法：
  python run_multi_asset.py
  python run_multi_asset.py --dir ~/Downloads/ --out ~/Documents/.../crypto/analysis/
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import run_all, scan_files_auto

# 排除项：key 在此集合中的文件会被跳过
EXCLUDE = {"DOGE_ema"}  # 回撤过大，默认排除

TITLE = "多标的策略分析"


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--out", default=os.path.expanduser(
                            "~/Documents/projects/quant-mh/crypto/analysis/"),
                        help="输出根目录")
    parser.add_argument("--window", type=int, default=12,
                        help="滚动窗口月数（默认 12）")
    args = parser.parse_args()

    print(f"扫描目录: {args.dir}")
    files = scan_files_auto(args.dir, exclude=EXCLUDE)

    if not files:
        print("未找到匹配的 xlsx 文件，请检查 --dir 路径或文件命名。")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：")
    for k, v in sorted(files.items()):
        print(f"  {k}: {Path(v).name}")

    run_all(files, title=TITLE, out_base=args.out, window=args.window)


if __name__ == "__main__":
    main()
