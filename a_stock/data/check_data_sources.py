"""
数据接口可用性验证脚本
检查以下接口是否可用、数据范围是否足够：
1. tushare index_dailybasic - 沪深300历史PE（市场状态区制用）
2. tushare index_weight - ETF成分股历史权重（行业拥挤度用）
3. akshare ETF资金流向历史数据

运行：cd a_stock/data && python check_data_sources.py
"""

import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch_data import init_pro

SAMPLE_ETF = "512760.SH"   # 半导体ETF，测试成分股接口
BENCHMARK  = "000300.SH"   # 沪深300
TEST_START = "20160101"
TEST_END   = "20160131"

print("=" * 60)
print("数据接口可用性验证")
print("=" * 60)

# ── 1. tushare index_dailybasic（市场PE/PB）─────────────────
print("\n[1] tushare index_dailybasic（沪深300 PE/PB）")
try:
    pro = init_pro()
    df = pro.index_dailybasic(ts_code=BENCHMARK, start_date=TEST_START, end_date=TEST_END,
                              fields="trade_date,pe_ttm,pb")
    if df is not None and not df.empty:
        print(f"    ✓ 可用，示例数据 {len(df)} 条")
        print(f"    字段：{list(df.columns)}")
        print(f"    示例：\n{df.head(3).to_string(index=False)}")
    else:
        print("    ✗ 返回空数据")
except Exception as e:
    print(f"    ✗ 报错：{e}")

# ── 2. tushare index_dailybasic 历史范围 ────────────────────
print("\n[2] index_dailybasic 历史数据范围（2016-2026全量）")
try:
    df_full = pro.index_dailybasic(ts_code=BENCHMARK, start_date="20160101", end_date="20261231",
                                   fields="trade_date,pe_ttm")
    if df_full is not None and not df_full.empty:
        df_full["trade_date"] = pd.to_datetime(df_full["trade_date"])
        print(f"    ✓ 共 {len(df_full)} 条，范围：{df_full['trade_date'].min().date()} ~ {df_full['trade_date'].max().date()}")
    else:
        print("    ✗ 返回空数据")
except Exception as e:
    print(f"    ✗ 报错：{e}")

# ── 3. tushare index_weight（ETF/指数成分股权重）────────────
print(f"\n[3] tushare index_weight（{SAMPLE_ETF} 成分股，单月）")
try:
    df_w = pro.index_weight(index_code=SAMPLE_ETF, start_date=TEST_START, end_date=TEST_END)
    if df_w is not None and not df_w.empty:
        print(f"    ✓ 可用，共 {len(df_w)} 条，成分股数 {df_w['con_code'].nunique()}")
        print(f"    字段：{list(df_w.columns)}")
        print(f"    示例：\n{df_w.head(3).to_string(index=False)}")
    else:
        print("    ✗ 返回空数据（可能权限不足或该ETF无成分股数据）")
except Exception as e:
    print(f"    ✗ 报错：{e}")

# ── 4. tushare fund_portfolio（ETF持仓，按季度）────────────
print(f"\n[4] tushare fund_portfolio（{SAMPLE_ETF} 基金持仓，季报）")
try:
    df_p = pro.fund_portfolio(ts_code=SAMPLE_ETF, start_date="20240101", end_date="20241231")
    if df_p is not None and not df_p.empty:
        print(f"    ✓ 可用，共 {len(df_p)} 条")
        print(f"    字段：{list(df_p.columns)}")
        print(f"    示例：\n{df_p.head(3).to_string(index=False)}")
    else:
        print("    ✗ 返回空数据")
except Exception as e:
    print(f"    ✗ 报错：{e}")

# ── 5. akshare ETF资金流向 ───────────────────────────────────
print(f"\n[5] akshare ETF资金流向（{SAMPLE_ETF.replace('.SH','')}）")
try:
    import akshare as ak
    # 尝试 fund_etf_fund_flow_hist 接口
    code_clean = SAMPLE_ETF.replace(".SH", "").replace(".SZ", "")
    df_flow = ak.fund_etf_fund_flow_hist(symbol=code_clean)
    if df_flow is not None and not df_flow.empty:
        print(f"    ✓ 可用，共 {len(df_flow)} 条")
        print(f"    字段：{list(df_flow.columns)}")
        df_flow_sorted = df_flow.sort_values(df_flow.columns[0])
        print(f"    数据范围：{df_flow_sorted.iloc[0, 0]} ~ {df_flow_sorted.iloc[-1, 0]}")
        print(f"    示例：\n{df_flow.head(3).to_string(index=False)}")
    else:
        print("    ✗ 返回空数据")
except ImportError:
    print("    ✗ akshare 未安装")
except Exception as e:
    print(f"    ✗ 报错：{e}")
    # 尝试备用接口
    print("    尝试备用接口 fund_etf_hist_em...")
    try:
        df_flow2 = ak.fund_etf_hist_em(symbol=code_clean, period="daily", start_date="20160101", end_date="20160131")
        if df_flow2 is not None and not df_flow2.empty:
            print(f"    ✓ fund_etf_hist_em 可用，字段：{list(df_flow2.columns)}")
        else:
            print("    ✗ 备用接口也返回空")
    except Exception as e2:
        print(f"    ✗ 备用接口报错：{e2}")

# ── 6. akshare ETF资金流向接口枚举 ──────────────────────────
print("\n[6] akshare 可用的ETF资金流相关接口")
try:
    import akshare as ak
    # 列出 akshare 中与 fund/etf/flow 相关的接口名
    flow_funcs = [name for name in dir(ak) if "fund" in name.lower() and ("flow" in name.lower() or "etf" in name.lower())]
    print(f"    找到 {len(flow_funcs)} 个相关接口：")
    for f in flow_funcs[:20]:
        print(f"      - {f}")
except ImportError:
    print("    ✗ akshare 未安装")
except Exception as e:
    print(f"    ✗ 报错：{e}")

print("\n" + "=" * 60)
print("验证完成")
