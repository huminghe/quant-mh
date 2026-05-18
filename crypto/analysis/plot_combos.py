"""
画各组合的收益曲线 + 回撤曲线
"""
import json, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open('/tmp/combo_data.json') as f:
    data = json.load(f)

dates = [datetime.date.fromisoformat(d) for d in data['dates']]
single = data['single']
combos = data['combos']

def drawdown_series(vals):
    """返回每个时间点的当前回撤（相对历史峰值的百分点下跌）"""
    peak = vals[0]
    dd = []
    for v in vals:
        if v > peak:
            peak = v
        dd.append(-(peak - v))  # 负值，方便图表显示
    return dd

def max_dd(vals):
    peak = vals[0]
    md = 0
    for v in vals:
        if v > peak:
            peak = v
        md = max(md, peak - v)
    return md

# ─── 颜色方案 ─────────────────────────────────────────────────────────────────
SINGLE_COLORS = {
    'BTC融合':  '#F7931A',
    'ETH融合':  '#627EEA',
    'SOL融合':  '#9945FF',
    'DOGE融合': '#C2A633',
}

# 两两组合：6种
PAIR_COLORS = {
    'ETH+SOL':  '#00B4D8',
    'BTC+SOL':  '#48CAE4',
    'BTC+ETH':  '#90E0EF',
    'SOL+DOGE': '#F4A261',
    'BTC+DOGE': '#E76F51',
    'ETH+DOGE': '#E9C46A',
}

# 三标的：4种
TRIPLE_COLORS = {
    'BTC+ETH+SOL':  '#2DC653',
    'ETH+SOL+DOGE': '#52B788',
    'BTC+SOL+DOGE': '#74C69D',
    'BTC+ETH+DOGE': '#95D5B2',
}

# 全组合
FULL_COLORS = {
    '全4标的':    '#E63946',
    '全11策略等权': '#1D3557',
}

# ═══════════════════════════════════════════════════════════════════════════════
# 图1：单标的融合 vs 两两组合（收益 + 回撤）
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.04,
    subplot_titles=['累计收益率 %', '回撤 %'],
)

# 单标的融合（细线，半透明）
for name, vals in single.items():
    color = SINGLE_COLORS[name]
    fig1.add_trace(go.Scatter(
        x=dates, y=vals,
        name=name,
        line=dict(color=color, width=1.5, dash='dot'),
        opacity=0.6,
        legendgroup='single',
        legendgrouptitle_text='单标的融合' if name == 'BTC融合' else None,
    ), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=dates, y=drawdown_series(vals),
        name=name, showlegend=False,
        line=dict(color=color, width=1, dash='dot'),
        opacity=0.5,
        legendgroup='single',
    ), row=2, col=1)

# 两两组合（粗线）
for name, vals in combos.items():
    if '+' not in name or name.count('+') != 1:
        continue
    color = PAIR_COLORS.get(name, '#888')
    md = max_dd(vals)
    fig1.add_trace(go.Scatter(
        x=dates, y=vals,
        name=f'{name} (DD:{md:.1f}%)',
        line=dict(color=color, width=2.5),
        legendgroup='pair',
        legendgrouptitle_text='两两组合' if name == 'ETH+SOL' else None,
    ), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=dates, y=drawdown_series(vals),
        name=name, showlegend=False,
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=color.replace('#', 'rgba(').replace(')', ',0.08)') if False else None,
        legendgroup='pair',
    ), row=2, col=1)

fig1.update_layout(
    title=dict(text='单标的融合 vs 两两组合 — 收益曲线 & 回撤', font=dict(size=15)),
    height=750,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(groupclick='toggleitem', tracegroupgap=8),
)
fig1.update_yaxes(ticksuffix='%', row=1, col=1)
fig1.update_yaxes(ticksuffix='%', row=2, col=1)
fig1.write_html('/Users/huminghe/Downloads/图4_两两组合收益回撤.html')
print('图4 完成')

# ═══════════════════════════════════════════════════════════════════════════════
# 图2：三标的组合 + 全组合（收益 + 回撤）
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.04,
    subplot_titles=['累计收益率 %', '回撤 %'],
)

# 单标的融合（背景参考，细虚线）
for name, vals in single.items():
    color = SINGLE_COLORS[name]
    fig2.add_trace(go.Scatter(
        x=dates, y=vals,
        name=name,
        line=dict(color=color, width=1, dash='dot'),
        opacity=0.4,
        legendgroup='single',
        legendgrouptitle_text='单标的（参考）' if name == 'BTC融合' else None,
    ), row=1, col=1)

# 三标的组合
for name, vals in combos.items():
    if name.count('+') != 2:
        continue
    color = TRIPLE_COLORS.get(name, '#888')
    md = max_dd(vals)
    fig2.add_trace(go.Scatter(
        x=dates, y=vals,
        name=f'{name} (DD:{md:.1f}%)',
        line=dict(color=color, width=2.5),
        legendgroup='triple',
        legendgrouptitle_text='三标的组合' if name == 'BTC+ETH+SOL' else None,
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=dates, y=drawdown_series(vals),
        name=name, showlegend=False,
        line=dict(color=color, width=2),
        legendgroup='triple',
    ), row=2, col=1)

# 全组合
for name, vals in combos.items():
    if name not in FULL_COLORS:
        continue
    color = FULL_COLORS[name]
    md = max_dd(vals)
    fig2.add_trace(go.Scatter(
        x=dates, y=vals,
        name=f'{name} (DD:{md:.1f}%)',
        line=dict(color=color, width=3.5),
        legendgroup='full',
        legendgrouptitle_text='全组合' if name == '全4标的' else None,
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=dates, y=drawdown_series(vals),
        name=name, showlegend=False,
        line=dict(color=color, width=3),
        fill='tozeroy',
        legendgroup='full',
    ), row=2, col=1)

fig2.update_layout(
    title=dict(text='三标的组合 & 全组合 — 收益曲线 & 回撤', font=dict(size=15)),
    height=750,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(groupclick='toggleitem', tracegroupgap=8),
)
fig2.update_yaxes(ticksuffix='%', row=1, col=1)
fig2.update_yaxes(ticksuffix='%', row=2, col=1)
fig2.write_html('/Users/huminghe/Downloads/图5_三标的全组合收益回撤.html')
print('图5 完成')

# ═══════════════════════════════════════════════════════════════════════════════
# 图3：所有组合一张图（只看收益，图例可点击开关）
# ═══════════════════════════════════════════════════════════════════════════════
fig3 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.04,
    subplot_titles=['累计收益率 %（所有组合）', '回撤 %'],
)

all_series = {}
all_series.update({k: v for k, v in single.items()})
all_series.update(combos)

# 按最大回撤排序，回撤小的在前
sorted_items = sorted(all_series.items(), key=lambda x: max_dd(x[1]))

color_map = {}
color_map.update(SINGLE_COLORS)
color_map.update(PAIR_COLORS)
color_map.update(TRIPLE_COLORS)
color_map.update(FULL_COLORS)

import colorsys
def auto_color(i, n):
    h = i / n
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

for i, (name, vals) in enumerate(sorted_items):
    color = color_map.get(name, auto_color(i, len(sorted_items)))
    md = max_dd(vals)
    is_full = name in FULL_COLORS
    width = 3.5 if is_full else (2.5 if '+' in name else 1.5)
    dash = 'dot' if name in SINGLE_COLORS else 'solid'

    fig3.add_trace(go.Scatter(
        x=dates, y=vals,
        name=f'{name}  DD:{md:.1f}%',
        line=dict(color=color, width=width, dash=dash),
        opacity=0.5 if name in SINGLE_COLORS else 1.0,
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=dates, y=drawdown_series(vals),
        name=name, showlegend=False,
        line=dict(color=color, width=max(1, width-1), dash=dash),
        opacity=0.5 if name in SINGLE_COLORS else 0.85,
    ), row=2, col=1)

fig3.update_layout(
    title=dict(text='所有组合收益 & 回撤总览（图例可点击开关）', font=dict(size=15)),
    height=800,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        font=dict(size=11),
        tracegroupgap=4,
    ),
)
fig3.update_yaxes(ticksuffix='%', row=1, col=1)
fig3.update_yaxes(ticksuffix='%', row=2, col=1)
fig3.write_html('/Users/huminghe/Downloads/图6_所有组合总览.html')
print('图6 完成')
print('\n全部完成：')
print('  图4_两两组合收益回撤.html')
print('  图5_三标的全组合收益回撤.html')
print('  图6_所有组合总览.html')


# ─── 供 run_analysis.py 调用的接口 ────────────────────────────────────────────

def run(files: dict, base: str, out_dir: str):
    """files: {key: 绝对路径}，base 传空字符串"""
    import os
    from analysis_utils import read_equity_curve as _rec, interp_series, drawdown_series as _dds

    _assets = sorted(set(k.split("_")[0] for k in files))
    _strats_per: dict[str, list[str]] = {}
    for k in files:
        a, s = k.split("_", 1)
        _strats_per.setdefault(a, []).append(s)

    # 读取曲线
    _raw_curves: dict[str, list] = {}
    for k, fp in files.items():
        _raw_curves[k] = _rec(base + fp)

    # 统一时间轴
    _all_dates = sorted(set(d for pts in _raw_curves.values() for d, _ in pts))

    _series: dict[str, list[float]] = {}
    for k, pts in _raw_curves.items():
        _series[k] = interp_series(pts, _all_dates)

    # 各标的融合
    _merged: dict[str, list[float]] = {}
    for a in _assets:
        keys = [f"{a}_{s}" for s in _strats_per.get(a, []) if f"{a}_{s}" in _series]
        if keys:
            _merged[a] = [sum(_series[k][i] for k in keys) / len(keys)
                          for i in range(len(_all_dates))]

    # 跨标的组合
    def _avg(*arrs):
        n = len(arrs)
        return [sum(a[i] for a in arrs) / n for i in range(len(arrs[0]))]

    _combos: dict[str, list[float]] = {}
    _combos.update({f"{a}融合": v for a, v in _merged.items()})

    # 两两
    asset_list = list(_merged.keys())
    for i in range(len(asset_list)):
        for j in range(i + 1, len(asset_list)):
            a1, a2 = asset_list[i], asset_list[j]
            _combos[f"{a1}+{a2}"] = _avg(_merged[a1], _merged[a2])
    # 三标的
    for i in range(len(asset_list)):
        for j in range(i + 1, len(asset_list)):
            for k in range(j + 1, len(asset_list)):
                a1, a2, a3 = asset_list[i], asset_list[j], asset_list[k]
                _combos[f"{a1}+{a2}+{a3}"] = _avg(_merged[a1], _merged[a2], _merged[a3])
    # 全组合
    if len(asset_list) >= 4:
        _combos["全4标的"] = _avg(*[_merged[a] for a in asset_list])
    _combos["全策略等权"] = [sum(_series[k][i] for k in _series) / len(_series)
                             for i in range(len(_all_dates))]

    # 颜色映射
    _single_colors = {f"{a}融合": c for a, c in
                      {'BTC': '#F7931A', 'ETH': '#627EEA',
                       'SOL': '#9945FF', 'DOGE': '#C2A633'}.items()}
    _full_colors = {'全4标的': '#E63946', '全策略等权': '#1D3557'}

    def _make_combo_fig(combo_keys, title, fname, single_ref=True):
        _fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.04,
            subplot_titles=['累计收益率 %', '回撤 %'],
        )
        if single_ref:
            for name, vals in _combos.items():
                if name not in _single_colors:
                    continue
                color = _single_colors[name]
                _fig.add_trace(go.Scatter(
                    x=_all_dates, y=vals, name=name,
                    line=dict(color=color, width=1.5, dash='dot'), opacity=0.5,
                    legendgroup='single',
                    legendgrouptitle_text='单标的（参考）' if list(_single_colors.keys()).index(name) == 0 else None,
                ), row=1, col=1)
        for name in combo_keys:
            vals = _combos.get(name)
            if vals is None:
                continue
            color = _full_colors.get(name, '#2DC653')
            md = max_dd(vals)
            _fig.add_trace(go.Scatter(
                x=_all_dates, y=vals,
                name=f"{name} (DD:{md:.1f}%)",
                line=dict(color=color, width=2.5 if name not in _full_colors else 3.5),
                legendgroup='combo',
            ), row=1, col=1)
            _fig.add_trace(go.Scatter(
                x=_all_dates, y=_dds(vals),
                name=name, showlegend=False,
                line=dict(color=color, width=2),
                legendgroup='combo',
            ), row=2, col=1)
        _fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            height=750, template='plotly_white', hovermode='x unified',
        )
        _fig.update_yaxes(ticksuffix='%', row=1, col=1)
        _fig.update_yaxes(ticksuffix='%', row=2, col=1)
        _fig.write_html(os.path.join(out_dir, fname))
        print(f"  {fname} 完成")

    # 两两组合图
    pair_keys = [k for k in _combos if k.count('+') == 1 and '融合' not in k]
    _make_combo_fig(pair_keys, '两两组合收益 & 回撤', '图4_两两组合收益回撤.html')

    # 三标的 + 全组合图
    triple_keys = [k for k in _combos if k.count('+') == 2 and '融合' not in k]
    full_keys = [k for k in _combos if k in _full_colors]
    _make_combo_fig(triple_keys + full_keys,
                    '三标的组合 & 全组合收益 & 回撤', '图5_三标的全组合收益回撤.html')

    # 总览图
    _fig_all = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=['累计收益率 %（所有组合）', '回撤 %'],
    )
    import colorsys as _cs
    _all_items = sorted(_combos.items(), key=lambda x: max_dd(x[1]))
    for idx, (name, vals) in enumerate(_all_items):
        color = _single_colors.get(name) or _full_colors.get(name)
        if color is None:
            h = idx / len(_all_items)
            r2, g2, b2 = _cs.hsv_to_rgb(h, 0.7, 0.85)
            color = f'#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}'
        md = max_dd(vals)
        is_single = name in _single_colors
        width = 3.5 if name in _full_colors else (2.5 if '+' in name else 1.5)
        dash = 'dot' if is_single else 'solid'
        _fig_all.add_trace(go.Scatter(
            x=_all_dates, y=vals,
            name=f'{name}  DD:{md:.1f}%',
            line=dict(color=color, width=width, dash=dash),
            opacity=0.5 if is_single else 1.0,
        ), row=1, col=1)
        _fig_all.add_trace(go.Scatter(
            x=_all_dates, y=_dds(vals),
            name=name, showlegend=False,
            line=dict(color=color, width=max(1, width - 1), dash=dash),
            opacity=0.5 if is_single else 0.85,
        ), row=2, col=1)
    _fig_all.update_layout(
        title=dict(text='所有组合收益 & 回撤总览（图例可点击开关）', font=dict(size=15)),
        height=800, template='plotly_white', hovermode='x unified',
    )
    _fig_all.update_yaxes(ticksuffix='%', row=1, col=1)
    _fig_all.update_yaxes(ticksuffix='%', row=2, col=1)
    _fig_all.write_html(os.path.join(out_dir, '图6_所有组合总览.html'))
    print("  图6_所有组合总览.html 完成")
