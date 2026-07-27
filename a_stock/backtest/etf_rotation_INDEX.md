# ETF 轮动回测脚本索引

41 个 `etf_rotation_*.py` 脚本的分类与定位。结论详见 `a_stock/docs/research.md`；本文件只做代码导航，不重复研究内容。

## 活跃模块（`a_stock/backtest/` 根目录，8 个，互相 import，请勿移动）

| 文件 | 作用 |
|------|------|
| `etf_rotation.py` | Round 1 基线：动量评分/调仓/回测核心函数，被其余 7 个文件及 `signal_today.py` 依赖 |
| `etf_rotation_v16_signal_combo_ablation.py` | 三信号集成评分模块，被 v21/v22/v25 及生产脚本 `signal_shadow_ensemble*.py` 依赖 |
| `etf_rotation_v21_fixed_calendar_days.py` | 调仓锚点敏感性测试，被 v22 依赖 |
| `etf_rotation_v22_score_blend.py` | 锚点得分加权融合测试 |
| `etf_rotation_v23_universe_bias_test.py` | 标的池选择偏差实测（机械化候选池构建），被 v24/v25/v26 依赖 |
| `etf_rotation_v24_top100_liquidity.py` | 标的池 + Top100 流动性上限变体 |
| `etf_rotation_v25_universe_ensemble_backtest.py` | 标的池 × 信号增强交叉验证 |
| `etf_rotation_v26_sw_stratified_universe.py` | 申万行业分层候选池方案 |

## 已归档脚本（`a_stock/backtest/archive/`，33 个，一次性诊断/已有定论，无外部 import）

| 文件 | 对应轮次/主题 | 结论 |
|------|--------------|------|
| `etf_rotation_analysis.py` | Round 1 | Top N × 窗口原始参数网格 |
| `etf_rotation_v2_analysis.py` | 第二轮 | 早期迭代 |
| `etf_rotation_v3_analysis.py`, `etf_rotation_v3_diagnosis.py` | 第三轮 | LW/区制切换/北向资金禁买，30+ 方向全部无效 |
| `etf_rotation_v3b_crowding.py`, `etf_rotation_v3b_fullval.py` | 第三轮方向 B | 拥挤度过滤改进后一度验证有效（后因复权 bug 修复被 v3c 推翻） |
| `etf_rotation_v3c_lw_fullval.py` | 第三轮方向 A 复核 | LW 方向不通过稳健性验证 |
| `etf_rotation_52wh.py` | — | 52 周高点信号，组合表现远逊基线，未入池 |
| `etf_rotation_p2_validate.py` | 第五~六轮 | 标的去重/子行业集中度/成交量确认 |
| `etf_rotation_v4_newfilters.py` | 第七轮 | 社融弱月横截面压制 + ETF 资金净流量反向软过滤，均不采用 |
| `etf_rotation_v5_new_directions.py` | 第八轮 | 新方向探索 |
| `etf_rotation_v9_quarterly.py`, `etf_rotation_v9_voltarget.py`, `etf_rotation_v9_qdii_ic.py` | 第九轮 | 季度调仓有害；波动率目标不稳健；QDII 标的池维持保留 |
| `etf_rotation_v10_lw_convergence.py`, `etf_rotation_v10_voltarget_convergence.py` | 第十轮 | convergence skill 判定均未收敛 |
| `etf_rotation_v10_pbo_collect.py`, `etf_rotation_v10_pbo_result.py` | 第十轮 | PBO 用法修正（不可跨方向混合候选） |
| `etf_rotation_v11_stock_bond_yield_gap.py`, `etf_rotation_v11_turn_of_month.py` | 第十轮 | 股债收益差轮动、日历效应，均不采用 |
| `etf_rotation_v12_riskadj_grid_pbo.py` | 第十轮 | 核心参数网格 PBO=0.400，判定不过拟合 |
| `etf_rotation_v13_futures_basis_ic.py`, `etf_rotation_v13_skew_ic.py` | 第十一轮 | 期货基差、收益偏度因子 IC 不达标 |
| `etf_rotation_v14_crossmarket_ic.py`, `etf_rotation_v14_crossmarket_daily_strategy.py`, `etf_rotation_v14_style_ic.py` | 第十一轮 | 跨市场溢出（隔夜缺口不可交易）、风格轮动，排除 |
| `etf_rotation_v15_dispersion_regime.py` | 第十二轮 | 行业收益离散度 regime，年度同向占比不足，排除 |
| `etf_rotation_v15_weak_signal_ensemble.py`, `etf_rotation_v15_weak_signal_ensemble_backtest.py` | 第十二轮 | ML 弱信号集成，通过稳健性检验（已上线影子监控） |
| `etf_rotation_v17_new_signal_ic.py` | 第十三轮 | 两融余额/资金流聚合/利率 Beta 三新信号 IC 检验 |
| `etf_rotation_v18_signal_ablation.py` | 第十三轮 | 4 信号 15 子集消融，margin_balance 组合层面无增量 |
| `etf_rotation_v19_full_signal_ablation.py` | 第十三轮 | 6 信号 63 子集全量消融，验证未过筛信号确实无隐藏价值 |
| `etf_rotation_v20_yearly_breakdown.py` | 第十三轮 | 全局最优子集分年度拆解，识别时间结构性衰减，不采用 |

## 目录结构说明

- 归档脚本移动时已同步修正 `sys.path`（多一层 `.parent`）及 `results/`/`data/` 相对路径引用，均已验证可正常 import。
- `results/*.png`、`results/*.csv` 等历史输出文件保留在 `a_stock/backtest/results/`，未随脚本移动。
- 归档脚本之间及 `etf_rotation_v10_pbo_collect.py` 的 `runpy` 调用关系均在同一目录内，未受影响。
