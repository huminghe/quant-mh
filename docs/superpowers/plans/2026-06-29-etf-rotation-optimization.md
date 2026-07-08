# ETF 轮动策略优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ETF 轮动策略基础上增加大盘趋势过滤和波动率反比加权两个优化，目标夏普 ≥ 0.75。

**Architecture:** 在 `etf_rotation.py` 的 `run_backtest()` 函数中增加两个 bool 开关（`use_market_filter`、`use_ivol_weighting`），默认关闭保持原行为；在 `etf_rotation_analysis.py` 中增加 4 组对比回测（原始/+过滤/+波动率权重/两者都加）并输出对比表和净值图。

**Tech Stack:** Python, pandas, numpy, matplotlib

---

### Task 1：在 `run_backtest()` 中增加大盘趋势过滤参数

**Files:**
- Modify: `a_stock/backtest/etf_rotation.py`

背景：`run_backtest()` 目前签名是 `run_backtest(close, scores, rebal_dates, top_n, init_cash, commission, slippage)`。需要增加 `use_market_filter: bool = False` 参数，在每个调仓日判断沪深300是否在 200 日均线之上，不满足则跳过建仓（清仓后保持空仓）。

- [ ] **Step 1：在文件顶部增加 MA200 常量**

在 `etf_rotation.py` 第 28 行（`START_DATE` 那行）后插入：

```python
MARKET_FILTER_MA     = 200      # 大盘趋势过滤均线周期（交易日）
IVOL_WINDOW          = 20       # 波动率反比加权计算窗口（交易日）
```

- [ ] **Step 2：修改 `run_backtest()` 函数签名**

将函数签名从：
```python
def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION,
    slippage: float = SLIPPAGE,
) -> pd.DataFrame:
```
改为：
```python
def run_backtest(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    rebal_dates: list,
    top_n: int = TOP_N,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION,
    slippage: float = SLIPPAGE,
    use_market_filter: bool = False,
    use_ivol_weighting: bool = False,
) -> pd.DataFrame:
```

- [ ] **Step 3：在 `run_backtest()` 函数体开头预计算 MA200**

在函数体第一行 `cash = init_cash` 之前插入：

```python
    # 预计算大盘 MA200（用于趋势过滤）
    if use_market_filter and BENCHMARK in close.columns:
        benchmark_ma200 = close[BENCHMARK].rolling(MARKET_FILTER_MA).mean()
    else:
        benchmark_ma200 = None
```

- [ ] **Step 4：在调仓逻辑入口增加大盘过滤判断**

找到 `if date in rebal_set:` 后的第一行（获取 `day_scores` 之前），在获取 `day_scores` 之前插入以下逻辑。完整的调仓块改为：

```python
        if date in rebal_set:
            # 大盘趋势过滤：若沪深300 < MA200，清仓保持空仓
            if use_market_filter and benchmark_ma200 is not None:
                ma200_val = benchmark_ma200.get(date)
                bench_close = close[BENCHMARK].get(date) if BENCHMARK in close.columns else None
                market_in_trend = (
                    bench_close is not None
                    and not pd.isna(bench_close)
                    and ma200_val is not None
                    and not pd.isna(ma200_val)
                    and bench_close > ma200_val
                )
            else:
                market_in_trend = True  # 不过滤时默认允许建仓

            # 清仓不在目标中的持仓
            if not market_in_trend:
                # 大盘趋势不满足，清空全部持仓
                for code in list(holdings.keys()):
                    price = close.loc[date, code] if code in close.columns else None
                    if price and not pd.isna(price):
                        sell_price = price * (1 - slippage / 2)
                        proceeds = holdings[code] * sell_price * (1 - commission)
                        cash += proceeds
                    del holdings[code]
                continue  # 跳过本月建仓

            # 获取当日有效得分（用当日收盘前的信号，即当日分数已计算完毕）
            day_scores = scores.loc[date].dropna()
```

- [ ] **Step 5：手动运行验证过滤逻辑不破坏原有行为**

在项目根目录运行（默认参数，行为应与原来完全一致）：
```bash
cd /Users/huminghe/Documents/projects/quant-mh
python a_stock/backtest/etf_rotation.py
```
预期：输出与之前相同的回测结果（年化约 12.3%，夏普约 0.63）。若输出不一致说明改动影响了原逻辑，需要检查。

---

### Task 2：在 `run_backtest()` 中增加波动率反比加权

**Files:**
- Modify: `a_stock/backtest/etf_rotation.py`

背景：当前等权分配逻辑在 `per_alloc = port_value / n` 这一行。需要在 `use_ivol_weighting=True` 时改为按各标的过去 20 日波动率的倒数加权。

- [ ] **Step 1：替换等权分配逻辑为条件分支**

找到调仓块中：
```python
            # 等权分配到目标标的
            n = len(target_codes)
            per_alloc = port_value / n  # 每只目标金额（用调仓前净值）
```

替换为：

```python
            # 仓位分配（等权 or 波动率反比加权）
            n = len(target_codes)
            if use_ivol_weighting and n > 0:
                # 计算各标的过去 IVOL_WINDOW 日年化波动率
                vols = {}
                for code in target_codes:
                    series = close[code].dropna()
                    # 取调仓日之前 IVOL_WINDOW 个交易日
                    loc = series.index.get_loc(date) if date in series.index else -1
                    if loc >= IVOL_WINDOW:
                        ret = series.iloc[loc - IVOL_WINDOW: loc].pct_change().dropna()
                        vol = ret.std() * np.sqrt(252)
                        vols[code] = vol if vol > 0 else 1e-6
                    else:
                        vols[code] = 1e-6  # 数据不足时用极小值（等效等权）
                inv_vols = {c: 1.0 / v for c, v in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {c: inv_vols[c] / total_inv for c in target_codes}
            else:
                weights = {c: 1.0 / n for c in target_codes}
```

- [ ] **Step 2：将 `per_alloc` 替换为按 `weights` 分配**

找到买入循环中：
```python            for code in target_codes:
                price = close.loc[date, code] if code in close.columns else None
                if price is None or pd.isna(price):
                    continue
                buy_price = price * (1 + slippage / 2)
                target_value = per_alloc
```

将 `target_value = per_alloc` 替换为：
```python                target_value = port_value * weights[code]
```

- [ ] **Step 3：运行验证波动率加权不破坏等权模式**

```bash
cd /Users/huminghe/Documents/projects/quant-mh
python a_stock/backtest/etf_rotation.py
```
预期：输出与原来一致（`use_ivol_weighting` 默认 False）。

---

### Task 3：更新 `etf_rotation_analysis.py` 增加 4 组对比回测

**Files:**
- Modify: `a_stock/backtest/etf_rotation_analysis.py`

背景：当前 `etf_rotation_analysis.py` 只测试原始策略的参数敏感性和 IS/OOS。需要在文件末尾增加一个 4 组对比回测（使用最优参数 Top3/25日），输出对比表和净值曲线。

- [ ] **Step 1：修改 `run_backtest()` 调用以支持新参数**

`etf_rotation_analysis.py` 中有自己的 `run_backtest()` 副本（代码重复，但暂不重构）。同样修改其函数签名和函数体，方式与 Task 1/Task 2 完全相同：

函数签名改为：
```python
def run_backtest(close, scores, rebal_dates, top_n, init_cash=INIT_CASH,
                 use_market_filter=False, use_ivol_weighting=False):
```

函数体在 `cash = init_cash` 之前插入：
```python
    MARKET_FILTER_MA = 200
    IVOL_WINDOW = 20
    BENCHMARK_LOCAL = "510300.SH"
    if use_market_filter and BENCHMARK_LOCAL in close.columns:
        benchmark_ma200 = close[BENCHMARK_LOCAL].rolling(MARKET_FILTER_MA).mean()
    else:
        benchmark_ma200 = None
```

调仓块替换方式与 Task 1 Step 4 完全相同（市场过滤部分），加权分配替换方式与 Task 2 Step 1、Step 2 完全相同。

- [ ] **Step 2：在文件末尾追加 4 组对比测试代码**

在 `plt.show()` 之后追加（文件最后）：

```python
# ── Part 3: 4 组优化对比（使用最优参数）────────────────────

print(f"\n运行 4 组优化对比（Top{best_top_n}，窗口{best_window}日）...")

CONFIGS = [
    ("原始策略",         False, False),
    ("+大盘过滤",        True,  False),
    ("+波动率加权",      False, True),
    ("+过滤+波动率加权", True,  True),
]

scores_opt = score_cache[best_window]
rebal_opt  = rebal_dates_full

compare_rows = []
compare_navs = {}

for label, use_mf, use_iv in CONFIGS:
    nav_c = run_backtest(close, scores_opt, rebal_opt, best_top_n,
                         use_market_filter=use_mf, use_ivol_weighting=use_iv)
    s = calc_stats(nav_c)
    compare_rows.append({
        "配置":     label,
        "年化收益": f"{s['CAGR']*100:.1f}%",
        "夏普":     f"{s['Sharpe']:.2f}",
        "最大回撤": f"{s['MaxDD']*100:.1f}%",
        "Calmar":   f"{s['Calmar']:.2f}",
    })
    compare_navs[label] = nav_c

compare_df = pd.DataFrame(compare_rows).set_index("配置")
print("\n" + "=" * 65)
print("4 组优化对比（全样本）")
print("=" * 65)
print(compare_df.to_string())

# 对比净值曲线
fig2, ax = plt.subplots(figsize=(14, 6))
colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"]
for (label, _, _), color in zip(CONFIGS, colors):
    nav_c = compare_navs[label]
    ax.plot(nav_c.index, nav_c / INIT_CASH, label=label, color=color, linewidth=1.4)
ax.plot(bench_nav_full.index, bench_nav_full / INIT_CASH,
        color="#FF9800", linewidth=1.2, alpha=0.7, linestyle="--", label="沪深300买持")
ax.set_title(f"4 组策略对比净值曲线（Top{best_top_n}，窗口{best_window}日）")
ax.set_ylabel("净值")
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
plt.tight_layout()
fig2_path = out_dir / "etf_rotation_compare.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"\n对比图表已保存：{fig2_path}")
plt.show()

# IS/OOS 验证最优配置
best_label, best_mf, best_iv = max(
    CONFIGS, key=lambda x: calc_stats(
        run_backtest(close, scores_opt, rebal_opt, best_top_n,
                     use_market_filter=x[1], use_ivol_weighting=x[2])
    )["Sharpe"]
)
print(f"\n最高夏普配置：{best_label}")
print("运行 IS/OOS 验证...")

nav_best_is  = run_backtest(close_is, scores_is, rebal_is, best_top_n,
                             use_market_filter=best_mf, use_ivol_weighting=best_iv)
nav_best_oos = run_backtest(close_oos, scores_oos, rebal_oos, best_top_n,
                             use_market_filter=best_mf, use_ivol_weighting=best_iv)

s_is  = calc_stats(nav_best_is)
s_oos = calc_stats(nav_best_oos)
print(f"IS  夏普：{s_is['Sharpe']:.2f}，年化收益：{s_is['CAGR']*100:.1f}%，最大回撤：{s_is['MaxDD']*100:.1f}%")
print(f"OOS 夏普：{s_oos['Sharpe']:.2f}，年化收益：{s_oos['CAGR']*100:.1f}%，最大回撤：{s_oos['MaxDD']*100:.1f}%")

decay = s_oos["Sharpe"] / s_is["Sharpe"] if s_is["Sharpe"] > 0 else 0
print(f"OOS/IS 夏普比：{decay:.2f}（>0.5 为可接受）")
if decay < 0.5:
    print("警告：OOS 夏普 < IS 夏普 × 0.5，可能存在过拟合")
else:
    print("通过：OOS 表现未显著衰减")
```

- [ ] **Step 3：运行完整分析脚本验证输出**

```bash
cd /Users/huminghe/Documents/projects/quant-mh
python a_stock/backtest/etf_rotation_analysis.py
```

预期输出：
1. 原有参数敏感性表（12 组）正常输出
2. IS/OOS 原始结果正常输出
3. 新增 4 组对比表，格式如下（数字仅供参考）：
```
配置                年化收益  夏普   最大回撤  Calmar
原始策略            12.x%    0.6x  -xx.x%   x.xx
+大盘过滤           xx.x%    0.7x  -xx.x%   x.xx
+波动率加权         xx.x%    0.6x  -xx.x%   x.xx
+过滤+波动率加权    xx.x%    0.7x  -xx.x%   x.xx
```
4. `a_stock/backtest/results/etf_rotation_compare.png` 图表文件生成

若运行报错，优先检查：
- `scores_opt` / `score_cache` 变量是否在追加代码之前已定义（在 Part 1 里定义）
- `close_is` / `close_oos` / `scores_is` / `scores_oos` 是否在追加代码之前已定义（在 Part 2 里定义）
