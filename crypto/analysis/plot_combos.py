"""
画各组合的收益曲线 + 回撤曲线
"""
import datetime
import colorsys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis_utils import read_equity_curve, interp_series, drawdown_series, max_drawdown


# ─── 颜色方案 ─────────────────────────────────────────────────────────────────
SINGLE_COLORS = {
    "BTC融合":  "#F7931A",
    "ETH融合":  "#627EEA",
    "SOL融合":  "#9945FF",
    "DOGE融合": "#C2A633",
}
FULL_COLORS = {"全4标的": "#E63946", "全策略等权": "#1D3557"}


def max_dd(vals):
    return max_drawdown(vals)


def auto_color(i, n):
    h = i / n
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"



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

    # 单标的模式：所有 key 属于同一标的，直接把每个版本当独立单元做组合
    _single_asset_mode = len(_assets) == 1
    if _single_asset_mode:
        # 每个版本作为独立曲线（用版本名作 key，去掉标的前缀）
        _units: dict[str, list[float]] = {}
        for k, vals in _series.items():
            _, strat = k.split("_", 1)
            _units[strat] = vals
        _combos.update(_units)
        unit_list = list(_units.keys())
        # 两两版本组合
        for i in range(len(unit_list)):
            for j in range(i + 1, len(unit_list)):
                u1, u2 = unit_list[i], unit_list[j]
                _combos[f"{u1}+{u2}"] = _avg(_units[u1], _units[u2])
        # 三版本组合
        for i in range(len(unit_list)):
            for j in range(i + 1, len(unit_list)):
                for kk in range(j + 1, len(unit_list)):
                    u1, u2, u3 = unit_list[i], unit_list[j], unit_list[kk]
                    _combos[f"{u1}+{u2}+{u3}"] = _avg(_units[u1], _units[u2], _units[u3])
        # 全版本等权
        if len(unit_list) >= 4:
            _combos["全版本等权"] = _avg(*list(_units.values()))
        _combos["全策略等权"] = [sum(_series[k][i] for k in _series) / len(_series)
                                 for i in range(len(_all_dates))]
    else:
        # 多标的模式：先按标的融合，再跨标的组合
        _combos.update({f"{a}融合": v for a, v in _merged.items()})
        asset_list = list(_merged.keys())
        # 两两
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
    _full_colors = {'全4标的': '#E63946', '全策略等权': '#1D3557', '全版本等权': '#1D3557'}

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
                    legendgroup=name,
                ), row=1, col=1)
                _fig.add_trace(go.Scatter(
                    x=_all_dates, y=_dds(vals),
                    name=name, showlegend=False,
                    line=dict(color=color, width=1, dash='dot'), opacity=0.5,
                    legendgroup=name,
                ), row=2, col=1)

        # 为 combo_keys 自动分配颜色（全4标的/全策略等权用固定色，其余 HSV 均匀分布）
        import colorsys as _cs
        non_fixed = [k for k in combo_keys if k not in _full_colors]
        _combo_colors = {}
        for idx, k in enumerate(non_fixed):
            h = idx / max(len(non_fixed), 1)
            r2, g2, b2 = _cs.hsv_to_rgb(h, 0.75, 0.80)
            _combo_colors[k] = f'#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}'
        _combo_colors.update(_full_colors)

        for name in combo_keys:
            vals = _combos.get(name)
            if vals is None:
                continue
            color = _combo_colors.get(name, '#888888')
            md = max_dd(vals)
            display_name = f"{name} (DD:{md:.1f}%)"
            _fig.add_trace(go.Scatter(
                x=_all_dates, y=vals,
                name=display_name,
                line=dict(color=color, width=3.5 if name in _full_colors else 2.5),
                legendgroup=name,
            ), row=1, col=1)
            _fig.add_trace(go.Scatter(
                x=_all_dates, y=_dds(vals),
                name=display_name, showlegend=False,
                line=dict(color=color, width=2),
                legendgroup=name,
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
    pair_title = '两两版本组合收益 & 回撤' if _single_asset_mode else '两两组合收益 & 回撤'
    _make_combo_fig(pair_keys, pair_title, '图4_两两组合收益回撤.html')

    # 三版本/三标的 + 全组合图
    triple_keys = [k for k in _combos if k.count('+') == 2 and '融合' not in k]
    full_keys = [k for k in _combos if k in _full_colors]
    triple_title = ('三版本组合 & 全版本等权收益 & 回撤' if _single_asset_mode
                    else '三标的组合 & 全组合收益 & 回撤')
    _make_combo_fig(triple_keys + full_keys, triple_title, '图5_三标的全组合收益回撤.html')

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
            legendgroup=name,
        ), row=1, col=1)
        _fig_all.add_trace(go.Scatter(
            x=_all_dates, y=_dds(vals),
            name=f'{name}  DD:{md:.1f}%', showlegend=False,
            line=dict(color=color, width=max(1, width - 1), dash=dash),
            opacity=0.5 if is_single else 0.85,
            legendgroup=name,
        ), row=2, col=1)
    _fig_all.update_layout(
        title=dict(text='所有组合收益 & 回撤总览（图例可点击开关）', font=dict(size=15)),
        height=800, template='plotly_white', hovermode='x unified',
    )
    _fig_all.update_yaxes(ticksuffix='%', row=1, col=1)
    _fig_all.update_yaxes(ticksuffix='%', row=2, col=1)
    _fig_all.write_html(os.path.join(out_dir, '图6_所有组合总览.html'))
    print("  图6_所有组合总览.html 完成")
