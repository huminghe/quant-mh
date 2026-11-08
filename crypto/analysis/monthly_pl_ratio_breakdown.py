"""
各策略逐月胜率/盈亏比明细（非滚动窗口，window=1）

rolling_pl_ratio.py 默认按滚动3个月窗口汇总，且只打印每个策略"最新窗口"一行，
看不到完整逐月序列。本脚本复用其 read_trade_pnls/monthly_win_loss/rolling_pl_ratio
函数，用 window=1 逐月拆分，打印每个策略最近 N 个月的胜率和盈亏比。

注意：样本数<10的月份数据噪音大，当月（尤其月初几天）样本极少，不能当结论看。

用法：
  python monthly_pl_ratio_breakdown.py                  # 扫描 ~/Downloads 最新 xlsx，默认最近12个月
  python monthly_pl_ratio_breakdown.py --months 24
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis_core import scan_files_auto
from rolling_pl_ratio import read_trade_pnls, monthly_win_loss, rolling_pl_ratio


def main():
    parser = argparse.ArgumentParser(description="各策略逐月胜率/盈亏比明细")
    parser.add_argument("--dir", default=os.path.expanduser("~/Downloads/"),
                        help="xlsx 文件所在目录（默认 ~/Downloads/）")
    parser.add_argument("--months", type=int, default=12,
                        help="展示最近N个月（默认12）")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="样本数低于该值标记为噪音（默认10）")
    args = parser.parse_args()

    files = scan_files_auto(args.dir)
    if not files:
        print(f"在 {args.dir} 未找到策略 xlsx 文件")
        sys.exit(1)

    for key in sorted(files.keys()):
        trades = read_trade_pnls(files[key])
        monthly = monthly_win_loss(trades)
        rows = rolling_pl_ratio(monthly, window=1)

        print(f"===== {key} =====")
        print(f"{'月份':<10} {'盈亏比':>8} {'胜率%':>8} {'样本数':>8} {'备注':<10}")
        for row in rows[-args.months:]:
            ym = row["end_month"]
            month_str = f"{ym[0]}-{ym[1]:02d}"
            note = "样本不足" if row["n_trades"] < args.min_trades else ""
            print(f"{month_str:<10} {row['pl_ratio']:>8.2f} {row['win_rate']:>7.1f}% "
                  f"{row['n_trades']:>8} {note:<10}")
        print()


if __name__ == "__main__":
    main()
