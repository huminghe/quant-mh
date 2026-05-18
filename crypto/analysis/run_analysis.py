"""
一键分析入口
用法：
  python run_analysis.py                        # 自动扫描 ~/Downloads/ 最新文件
  python run_analysis.py --dir /path/to/xlsx    # 指定目录
  python run_analysis.py --out /path/to/output  # 指定输出目录

自动按文件名规则匹配标的和策略版本：
  strategy_ema_{asset}_*  → {ASSET}_ema
  v2_strategy_{asset}_*   → {ASSET}_v2
  v3_3h_strategy_{asset}_* / v3_205m_strategy_{asset}_* → {ASSET}_v3

排除 DOGE_ema（回撤过大，自动跳过）。
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

import analyze_strategies
import plot_equity_curves
import plot_combos
import rolling_analysis
import heatmap_analysis
import correlation_matrix
import generate_detail_report


# ─── 文件名匹配规则 ────────────────────────────────────────────────────────────

ASSET_ALIASES = {
    "btc":  "BTC",
    "eth":  "ETH",
    "sol":  "SOL",
    "doge": "DOGE",
    "meme": "DOGE",   # strategy_ema_meme_... → DOGE
}

STRAT_PATTERNS = [
    (r"^strategy_ema_",      "ema"),
    (r"^v2_strategy_",       "v2"),
    (r"^v3_3h_strategy_",    "v3"),
    (r"^v3_205m_strategy_",  "v3"),
]

EXCLUDE = {"DOGE_ema"}   # 固定排除


def scan_files(directory: str) -> dict[str, str]:
    """
    扫描目录，找出最新的一批 xlsx 文件（按日期分组，取最新日期）。
    返回 {key: filepath}，key 格式为 ASSET_strat。
    """
    d = Path(directory)
    xlsx_files = list(d.glob("*.xlsx"))
    if not xlsx_files:
        return {}

    # 从文件名提取日期（格式 _YYYY-MM-DD_）
    date_pattern = re.compile(r"_(\d{4}-\d{2}-\d{2})_")
    dated: list[tuple[datetime.date, Path]] = []
    for f in xlsx_files:
        m = date_pattern.search(f.name)
        if m:
            dated.append((datetime.date.fromisoformat(m.group(1)), f))

    if not dated:
        return {}

    # 取最新日期的文件
    latest_date = max(d for d, _ in dated)
    latest_files = [f for d, f in dated if d == latest_date]

    # 匹配标的和策略
    result: dict[str, str] = {}
    for f in latest_files:
        name = f.name.lower()

        # 匹配策略版本
        strat = None
        for pattern, s in STRAT_PATTERNS:
            if re.match(pattern, name):
                strat = s
                break
        if strat is None:
            continue

        # 匹配标的
        asset = None
        for alias, canonical in ASSET_ALIASES.items():
            # 文件名中包含 _{alias}_ 或 _{alias}usdt
            if f"_{alias}_" in name or f"_{alias}usdt" in name:
                asset = canonical
                break
        if asset is None:
            continue

        key = f"{asset}_{strat}"
        if key in EXCLUDE:
            print(f"  跳过（已排除）: {f.name}")
            continue

        # 同一 key 有多个文件时取最新
        if key not in result or f.name > Path(result[key]).name:
            result[key] = str(f)

    return result


def make_output_dir(base_dir: str, date_str: str) -> str:
    """创建带日期的输出目录。"""
    out = Path(base_dir) / f"charts_{date_str}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="加密货币策略一键分析")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认与 --dir 相同）")
    parser.add_argument("--window", type=int, default=12,
                        help="滚动窗口月数（默认 12）")
    args = parser.parse_args()

    src_dir = args.dir.rstrip("/") + "/"
    out_base = args.out or args.dir

    # 1. 扫描文件
    print(f"扫描目录: {src_dir}")
    files = scan_files(src_dir)
    if not files:
        print("未找到匹配的 xlsx 文件，请检查目录或文件命名。")
        sys.exit(1)

    print(f"找到 {len(files)} 个策略文件：")
    for k, v in sorted(files.items()):
        print(f"  {k}: {Path(v).name}")

    # 2. 确定输出目录
    date_str = datetime.date.today().isoformat()
    out_dir = make_output_dir(out_base, date_str)
    print(f"\n输出目录: {out_dir}\n")

    # 文件路径转为 {key: filename}（analyze_strategies 等模块需要 base+fname）
    # 统一用绝对路径，base 设为空字符串
    abs_files = {k: v for k, v in files.items()}

    # 3. 各模块分析
    print("=" * 50)
    print("【1/5】生成指标汇总 Excel 报告...")
    try:
        analyze_strategies.run(abs_files, base="", out_dir=out_dir,
                               date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【2/5】生成收益走势图...")
    try:
        plot_equity_curves.run(abs_files, base="", out_dir=out_dir)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【3/5】生成组合收益回撤图...")
    try:
        plot_combos.run(abs_files, base="", out_dir=out_dir)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【4/5】生成滚动窗口分析...")
    try:
        rolling_analysis.run(abs_files, base="", out_dir=out_dir,
                             window=args.window)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【5/6】生成热力图 & 相关性矩阵...")
    try:
        heatmap_analysis.run(abs_files, base="", out_dir=out_dir)
    except Exception as e:
        print(f"  警告: {e}")
    try:
        correlation_matrix.run(abs_files, base="", out_dir=out_dir)
    except Exception as e:
        print(f"  警告: {e}")

    print("\n【6/6】生成详细分析 Excel 报告（相关性+滚动+月度明细）...")
    try:
        generate_detail_report.run(abs_files, base="", out_dir=out_dir,
                                   date_str=date_str)
    except Exception as e:
        print(f"  警告: {e}")

    print(f"\n全部完成！输出目录: {out_dir}")
    print("文件列表:")
    for f in sorted(Path(out_dir).iterdir()):
        size = f.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        size_str = f"{size/1024:.0f}{unit}" if size < 1_000_000 else f"{size/1_000_000:.1f}{unit}"
        print(f"  {f.name}  ({size_str})")


if __name__ == "__main__":
    main()
