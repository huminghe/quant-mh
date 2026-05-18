"""
策略综合分析脚本
读取 BTC/ETH/SOL/DOGE 四个标的、三种策略的回测数据
生成：1) 指标对比表  2) 融合分析  3) 收益走势图
"""
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              numbers as xl_numbers)
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.series import SeriesLabel
import datetime
import os

BASE = '/Users/huminghe/Downloads/'

FILES = {
    'BTC': {
        'ema':  'strategy_ema_btc_OKX_BTCUSDT.P_2026-05-18_bccdc.xlsx',
        'v2':   'v2_strategy_btc_OKX_BTCUSDT.P_2026-05-18_89f2b.xlsx',
        'v3':   'v3_205m_strategy_btc_OKX_BTCUSDT.P_2026-05-18_7f0bc.xlsx',
    },
    'ETH': {
        'ema':  'strategy_ema_eth_OKX_ETHUSDT.P_2026-05-18_6d950.xlsx',
        'v2':   'v2_strategy_eth_OKX_ETHUSDT.P_2026-05-18_7eba3.xlsx',
        'v3':   'v3_3h_strategy_eth_OKX_ETHUSDT.P_2026-05-18_f84bb.xlsx',
    },
    'SOL': {
        'ema':  'strategy_ema_sol_OKX_SOLUSDT.P_2026-05-18_f737f.xlsx',
        'v2':   'v2_strategy_sol_OKX_SOLUSDT.P_2026-05-18_2c3a0.xlsx',
        'v3':   'v3_3h_strategy_sol_OKX_SOLUSDT.P_2026-05-18_a51fb.xlsx',
    },
    'DOGE': {
        'ema':  'strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-18_3a4e0.xlsx',
        'v2':   'v2_strategy_doge_OKX_DOGEUSDT.P_2026-05-18_2b9fa.xlsx',
        'v3':   'v3_205m_strategy_doge_OKX_DOGEUSDT.P_2026-05-18_283df.xlsx',
    },
}

ASSETS = ['BTC', 'ETH', 'SOL', 'DOGE']
STRATS = ['ema', 'v2', 'v3']
STRAT_LABELS = {'ema': 'EMA', 'v2': 'V2', 'v3': 'V3'}

# ─── 样式 ────────────────────────────────────────────────────────────────────
HDR_FILL   = PatternFill('solid', start_color='1F4E79')
HDR_FONT   = Font(bold=True, color='FFFFFF', name='Arial', size=10)
SUB_FILL   = PatternFill('solid', start_color='2E75B6')
SUB_FONT   = Font(bold=True, color='FFFFFF', name='Arial', size=10)
BEST_FILL  = PatternFill('solid', start_color='E2EFDA')
WARN_FILL  = PatternFill('solid', start_color='FCE4D6')
ALT_FILL   = PatternFill('solid', start_color='F2F2F2')
BODY_FONT  = Font(name='Arial', size=10)
BOLD_FONT  = Font(bold=True, name='Arial', size=10)
CENTER     = Alignment(horizontal='center', vertical='center')
LEFT       = Alignment(horizontal='left',   vertical='center')
RIGHT      = Alignment(horizontal='right',  vertical='center')

THIN = Side(style='thin', color='BFBFBF')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

ASSET_COLORS = {
    'BTC':  'F7931A',
    'ETH':  '627EEA',
    'SOL':  '9945FF',
    'DOGE': 'C2A633',
}

def cell_style(ws, cell_ref, fill=None, font=None, align=None, border=None, num_fmt=None):
    c = ws[cell_ref] if isinstance(cell_ref, str) else cell_ref
    if fill:   c.fill      = fill
    if font:   c.font      = font
    if align:  c.alignment = align
    if border: c.border    = border
    if num_fmt: c.number_format = num_fmt

def set_row_style(ws, row, cols, fill=None, font=None, align=None, border=None):
    for col in range(1, cols + 1):
        c = ws.cell(row=row, column=col)
        if fill:   c.fill      = fill
        if font:   c.font      = font
        if align:  c.alignment = align
        if border: c.border    = border

# ─── 数据读取 ─────────────────────────────────────────────────────────────────
def read_stats(filepath):
    wb = openpyxl.load_workbook(filepath, read_only=True)
    stats = {}
    for sheet_name in ['表现', '交易分析', '风险调整后的表现']:
        ws = wb[sheet_name]
        for r in ws.iter_rows(values_only=True):
            if r[0] is not None:
                stats[r[0]] = {
                    '全部USDT': r[1], '全部%': r[2],
                    '多头USDT': r[3], '多头%': r[4],
                    '空头USDT': r[5], '空头%': r[6],
                }
    wb.close()
    return stats

def read_equity_curve(filepath):
    """返回 [(datetime, cumulative_pnl_pct), ...] 出场时间序列"""
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb['交易清单']
    points = []
    for r in ws.iter_rows(values_only=True):
        if r[1] and '出场' in str(r[1]) and isinstance(r[2], datetime.datetime) and r[14] is not None:
            points.append((r[2], float(r[14])))
    wb.close()
    return sorted(points, key=lambda x: x[0])

def get_key(stats, key, field='全部%', default=None):
    return stats.get(key, {}).get(field, default)

# ─── 加载所有数据 ─────────────────────────────────────────────────────────────
print("读取数据中...")
all_stats = {}
all_equity = {}
for asset in ASSETS:
    all_stats[asset] = {}
    all_equity[asset] = {}
    for strat in STRATS:
        fp = BASE + FILES[asset][strat]
        all_stats[asset][strat] = read_stats(fp)
        all_equity[asset][strat] = read_equity_curve(fp)
        print(f"  {asset} {strat}: {len(all_equity[asset][strat])} 笔出场")

print("数据读取完成，开始生成报告...")

# ─── 创建工作簿 ───────────────────────────────────────────────────────────────
wb_out = Workbook()
wb_out.remove(wb_out.active)  # 删除默认sheet

# ═══════════════════════════════════════════════════════════════════════════════
# Sheet 1: 总览对比表
# ═══════════════════════════════════════════════════════════════════════════════
ws1 = wb_out.create_sheet('总览对比')

METRICS = [
    ('净利润 %',        '净利润',           '全部%',    '%.2f%%',  True),
    ('年化收益 %',      '年化收益率（CAGR）','全部%',    '%.2f%%',  True),
    ('最大回撤 %',      '最大股权回撤（intrabar）','全部%','%.2f%%', False),
    ('夏普比率',        '夏普比率',          '全部USDT', '%.3f',    True),
    ('Sortino比率',     'Sortino比率',       '全部USDT', '%.3f',    True),
    ('盈利因子',        '盈利因子',          '全部USDT', '%.3f',    True),
    ('胜率 %',          '获利百分比',        '全部%',    '%.2f%%',  True),
    ('盈亏比',          '平均胜率/平均负率', '全部USDT', '%.3f',    True),
    ('总交易次数',      '总交易',            '全部USDT', '%d',      None),
    ('多头净利润 %',    '净利润',            '多头%',    '%.2f%%',  True),
    ('空头净利润 %',    '净利润',            '空头%',    '%.2f%%',  True),
    ('多头盈利因子',    '盈利因子',          '多头USDT', '%.3f',    True),
    ('空头盈利因子',    '盈利因子',          '空头USDT', '%.3f',    True),
]

# 标题行 — A列不合并，避免后续数据行写入 MergedCell
ws1['A1'] = '指标'
ws1['A2'] = ''
cell_style(ws1, 'A1', fill=HDR_FILL, font=HDR_FONT, align=CENTER, border=BORDER)
cell_style(ws1, 'A2', fill=HDR_FILL, font=HDR_FONT, align=CENTER, border=BORDER)

col = 2
asset_col_start = {}
for asset in ASSETS:
    asset_col_start[asset] = col
    end_col = col + len(STRATS) - 1
    ws1.merge_cells(start_row=1, start_column=col, end_row=1, end_column=end_col)
    c = ws1.cell(row=1, column=col, value=asset)
    c.fill = PatternFill('solid', start_color=ASSET_COLORS[asset])
    c.font = Font(bold=True, color='FFFFFF', name='Arial', size=11)
    c.alignment = CENTER
    c.border = BORDER
    for s_idx, strat in enumerate(STRATS):
        ws1.merge_cells(start_row=2, start_column=col+s_idx, end_row=2, end_column=col+s_idx)
        c2 = ws1.cell(row=2, column=col+s_idx, value=STRAT_LABELS[strat])
        c2.fill = SUB_FILL
        c2.font = SUB_FONT
        c2.alignment = CENTER
        c2.border = BORDER
    col += len(STRATS)

# 指标列标题
ws1['A1'].alignment = CENTER

# 数据行
for m_idx, (label, key, field, fmt, higher_better) in enumerate(METRICS):
    row = m_idx + 3
    c = ws1.cell(row=row, column=1, value=label)
    c.font = BOLD_FONT
    c.alignment = LEFT
    c.border = BORDER
    if row % 2 == 0:
        c.fill = ALT_FILL

    # 收集该指标所有值，用于高亮最优
    vals = {}
    for asset in ASSETS:
        for strat in STRATS:
            v = get_key(all_stats[asset][strat], key, field)
            if v is not None:
                vals[(asset, strat)] = float(v)

    if vals and higher_better is not None:
        best_val = max(vals.values()) if higher_better else min(vals.values())
        worst_val = min(vals.values()) if higher_better else max(vals.values())
    else:
        best_val = worst_val = None

    col = 2
    for asset in ASSETS:
        for strat in STRATS:
            v = get_key(all_stats[asset][strat], key, field)
            cell = ws1.cell(row=row, column=col)
            if v is not None:
                fv = float(v)
                if fmt == '%d':
                    cell.value = int(fv)
                else:
                    cell.value = fv
                    if '%' in fmt:
                        cell.number_format = '0.00"%"'
                    else:
                        cell.number_format = '0.000'
                # 高亮
                if best_val is not None and abs(fv - best_val) < 1e-9:
                    cell.fill = BEST_FILL
                    cell.font = Font(bold=True, name='Arial', size=10, color='375623')
                elif worst_val is not None and abs(fv - worst_val) < 1e-9 and best_val != worst_val:
                    cell.fill = WARN_FILL
                    cell.font = Font(name='Arial', size=10, color='9C0006')
                else:
                    cell.font = BODY_FONT
                    if row % 2 == 0:
                        cell.fill = ALT_FILL
            else:
                cell.value = '-'
                cell.font = BODY_FONT
            cell.alignment = CENTER
            cell.border = BORDER
            col += 1

# 列宽
ws1.column_dimensions['A'].width = 20
for col_idx in range(2, 2 + len(ASSETS) * len(STRATS)):
    ws1.column_dimensions[get_column_letter(col_idx)].width = 10
ws1.row_dimensions[1].height = 22
ws1.row_dimensions[2].height = 18
ws1.freeze_panes = 'B3'

print("  Sheet1 总览对比 完成")

# ═══════════════════════════════════════════════════════════════════════════════
# Sheet 2: 各标的融合分析
# ═══════════════════════════════════════════════════════════════════════════════
ws2 = wb_out.create_sheet('融合分析')

# 融合方案：等权平均三策略的累计P&L%
# 先把每个策略的时间序列插值到统一时间轴，再平均

def interpolate_equity(points, dates):
    """将稀疏的 (date, pnl%) 序列插值到 dates 列表"""
    if not points:
        return [0.0] * len(dates)
    result = []
    j = 0
    for d in dates:
        while j < len(points) - 1 and points[j+1][0] <= d:
            j += 1
        result.append(points[j][1] if points[j][0] <= d else 0.0)
    return result

# 收集所有日期
all_dates_set = set()
for asset in ASSETS:
    for strat in STRATS:
        for dt, _ in all_equity[asset][strat]:
            all_dates_set.add(dt.date())
all_dates = sorted(all_dates_set)

# 计算每个标的的融合曲线（等权三策略）
merged_equity = {}
for asset in ASSETS:
    curves = []
    for strat in STRATS:
        pts = [(p[0].date() if isinstance(p[0], datetime.datetime) else p[0], p[1])
               for p in all_equity[asset][strat]]
        # 按日期去重取最后值
        date_map = {}
        for d, v in pts:
            date_map[d] = v
        sorted_pts = sorted(date_map.items())
        curves.append(interpolate_equity(sorted_pts, all_dates))
    # 等权平均
    merged = [(all_dates[i], (curves[0][i] + curves[1][i] + curves[2][i]) / 3.0)
              for i in range(len(all_dates))]
    merged_equity[asset] = merged

# 全组合融合（4标的等权）
portfolio_equity = []
for i, d in enumerate(all_dates):
    avg = sum(merged_equity[asset][i][1] for asset in ASSETS) / len(ASSETS)
    portfolio_equity.append((d, avg))

# 写入融合分析sheet
ws2.merge_cells('A1:I1')
ws2['A1'] = '各标的策略融合分析（三策略等权平均）'
ws2['A1'].font = Font(bold=True, name='Arial', size=13, color='1F4E79')
ws2['A1'].alignment = CENTER
ws2['A1'].fill = PatternFill('solid', start_color='D6E4F0')

# 融合指标汇总表
headers2 = ['标的', '融合净利润%', '融合年化%', '最优策略', '最优净利润%', '最优夏普', '最优回撤%', '推荐配置']
row2 = 3
for col_idx, h in enumerate(headers2, 1):
    c = ws2.cell(row=row2, column=col_idx, value=h)
    c.fill = HDR_FILL
    c.font = HDR_FONT
    c.alignment = CENTER
    c.border = BORDER

for a_idx, asset in enumerate(ASSETS):
    row2 += 1
    # 融合净利润 = 三策略等权平均最终值
    final_merged = merged_equity[asset][-1][1] if merged_equity[asset] else 0
    # 找最优策略（按夏普）
    best_strat = max(STRATS, key=lambda s: get_key(all_stats[asset][s], '夏普比率', '全部USDT') or 0)
    best_net = get_key(all_stats[asset][best_strat], '净利润', '全部%') or 0
    best_sharpe = get_key(all_stats[asset][best_strat], '夏普比率', '全部USDT') or 0
    best_dd = get_key(all_stats[asset][best_strat], '最大股权回撤（intrabar）', '全部%') or 0
    # 年化（用最优策略的年化）
    best_cagr = get_key(all_stats[asset][best_strat], '年化收益率（CAGR）', '全部%') or 0

    # 推荐配置逻辑
    if best_sharpe >= 0.4:
        rec = '重点配置'
    elif best_sharpe >= 0.25:
        rec = '标准配置'
    else:
        rec = '轻仓/观察'

    row_data = [asset, final_merged, best_cagr, STRAT_LABELS[best_strat],
                best_net, best_sharpe, best_dd, rec]
    for col_idx, val in enumerate(row_data, 1):
        c = ws2.cell(row=row2, column=col_idx, value=val)
        c.font = BODY_FONT
        c.alignment = CENTER
        c.border = BORDER
        if a_idx % 2 == 0:
            c.fill = ALT_FILL
        if col_idx in (2, 3, 5):
            c.number_format = '0.00"%"'
        elif col_idx == 6:
            c.number_format = '0.000'
        elif col_idx == 7:
            c.number_format = '0.00"%"'
        # 颜色标注推荐
        if col_idx == 8:
            if val == '重点配置':
                c.fill = BEST_FILL
                c.font = Font(bold=True, name='Arial', size=10, color='375623')
            elif val == '轻仓/观察':
                c.fill = WARN_FILL
                c.font = Font(name='Arial', size=10, color='9C0006')

# 列宽
for col_idx, w in enumerate([10, 14, 12, 12, 14, 10, 12, 12], 1):
    ws2.column_dimensions[get_column_letter(col_idx)].width = w

print("  Sheet2 融合分析 完成")

# ═══════════════════════════════════════════════════════════════════════════════
# Sheet 3: 收益走势数据（供图表用）
# ═══════════════════════════════════════════════════════════════════════════════
ws3 = wb_out.create_sheet('收益走势数据')

# 写表头
headers3 = ['日期'] + [f'{a}_{s}' for a in ASSETS for s in STRATS] + \
           [f'{a}_融合' for a in ASSETS] + ['全组合']
for col_idx, h in enumerate(headers3, 1):
    c = ws3.cell(row=1, column=col_idx, value=h)
    c.fill = HDR_FILL
    c.font = HDR_FONT
    c.alignment = CENTER
    c.border = BORDER

# 预计算每个策略的日期→值映射
strat_date_map = {}
for asset in ASSETS:
    for strat in STRATS:
        pts = all_equity[asset][strat]
        dm = {}
        for dt, v in pts:
            d = dt.date() if isinstance(dt, datetime.datetime) else dt
            dm[d] = v
        strat_date_map[(asset, strat)] = dm

# 写数据行（每月取一个点，减少行数）
monthly_dates = []
prev_month = None
for d in all_dates:
    ym = (d.year, d.month)
    if ym != prev_month:
        monthly_dates.append(d)
        prev_month = ym

def get_val_at(date_map, d):
    """取 d 当天或之前最近的值"""
    best = None
    for dd, v in sorted(date_map.items()):
        if dd <= d:
            best = v
    return best or 0.0

for row_idx, d in enumerate(monthly_dates, 2):
    ws3.cell(row=row_idx, column=1, value=datetime.datetime(d.year, d.month, d.day))
    ws3.cell(row=row_idx, column=1).number_format = 'YYYY-MM'
    col = 2
    for asset in ASSETS:
        for strat in STRATS:
            v = get_val_at(strat_date_map[(asset, strat)], d)
            ws3.cell(row=row_idx, column=col, value=round(v, 2))
            col += 1
    for asset in ASSETS:
        v = get_val_at({dd: vv for dd, vv in merged_equity[asset]}, d)
        ws3.cell(row=row_idx, column=col, value=round(v, 2))
        col += 1
    # 全组合
    v = get_val_at({dd: vv for dd, vv in portfolio_equity}, d)
    ws3.cell(row=row_idx, column=col, value=round(v, 2))

ws3.column_dimensions['A'].width = 12
for col_idx in range(2, len(headers3) + 1):
    ws3.column_dimensions[get_column_letter(col_idx)].width = 10
ws3.freeze_panes = 'B2'

print("  Sheet3 收益走势数据 完成")

# ═══════════════════════════════════════════════════════════════════════════════
# Sheet 4-7: 各标的详细分析（每个标的一个sheet）
# ═══════════════════════════════════════════════════════════════════════════════
DETAIL_METRICS = [
    ('净利润 %',        '净利润',           '全部%'),
    ('多头净利润 %',    '净利润',           '多头%'),
    ('空头净利润 %',    '净利润',           '空头%'),
    ('年化收益 %',      '年化收益率（CAGR）','全部%'),
    ('最大回撤 %',      '最大股权回撤（intrabar）','全部%'),
    ('夏普比率',        '夏普比率',          '全部USDT'),
    ('Sortino比率',     'Sortino比率',       '全部USDT'),
    ('盈利因子',        '盈利因子',          '全部USDT'),
    ('多头盈利因子',    '盈利因子',          '多头USDT'),
    ('空头盈利因子',    '盈利因子',          '空头USDT'),
    ('胜率 %',          '获利百分比',        '全部%'),
    ('盈亏比',          '平均胜率/平均负率', '全部USDT'),
    ('总交易次数',      '总交易',            '全部USDT'),
    ('平均盈利交易',    '平均盈利交易',      '全部USDT'),
    ('平均亏损交易',    '平均亏损交易',      '全部USDT'),
    ('净利润/最大回撤', '净利润占最大亏损的百分比', '全部%'),
]

for asset in ASSETS:
    ws = wb_out.create_sheet(f'{asset}详情')
    # 标题
    ws.merge_cells('A1:D1')
    ws['A1'] = f'{asset} — 三策略详细对比'
    ws['A1'].font = Font(bold=True, name='Arial', size=13,
                         color=ASSET_COLORS[asset])
    ws['A1'].alignment = CENTER
    ws['A1'].fill = PatternFill('solid', start_color='F2F2F2')

    # 表头
    for col_idx, h in enumerate(['指标', 'EMA', 'V2', 'V3'], 1):
        c = ws.cell(row=2, column=col_idx, value=h)
        c.fill = HDR_FILL
        c.font = HDR_FONT
        c.alignment = CENTER
        c.border = BORDER

    for m_idx, (label, key, field) in enumerate(DETAIL_METRICS):
        row = m_idx + 3
        c = ws.cell(row=row, column=1, value=label)
        c.font = BOLD_FONT
        c.alignment = LEFT
        c.border = BORDER
        if row % 2 == 0:
            c.fill = ALT_FILL

        vals = {}
        for s_idx, strat in enumerate(STRATS):
            v = get_key(all_stats[asset][strat], key, field)
            if v is not None:
                vals[strat] = float(v)

        # 判断越大越好
        higher_better_map = {
            '净利润 %': True, '多头净利润 %': True, '空头净利润 %': True,
            '年化收益 %': True, '最大回撤 %': False,
            '夏普比率': True, 'Sortino比率': True, '盈利因子': True,
            '多头盈利因子': True, '空头盈利因子': True,
            '胜率 %': True, '盈亏比': True, '总交易次数': None,
            '平均盈利交易': True, '平均亏损交易': False,
            '净利润/最大回撤': True,
        }
        hb = higher_better_map.get(label)
        if vals and hb is not None:
            best_v = max(vals.values()) if hb else min(vals.values())
        else:
            best_v = None

        for s_idx, strat in enumerate(STRATS):
            col = s_idx + 2
            v = vals.get(strat)
            c = ws.cell(row=row, column=col)
            if v is not None:
                c.value = v
                if label in ('总交易次数',):
                    c.number_format = '0'
                elif '%' in label:
                    c.number_format = '0.00"%"'
                else:
                    c.number_format = '0.000'
                if best_v is not None and abs(v - best_v) < 1e-9:
                    c.fill = BEST_FILL
                    c.font = Font(bold=True, name='Arial', size=10, color='375623')
                else:
                    c.font = BODY_FONT
                    if row % 2 == 0:
                        c.fill = ALT_FILL
            else:
                c.value = '-'
                c.font = BODY_FONT
            c.alignment = CENTER
            c.border = BORDER

    ws.column_dimensions['A'].width = 22
    for col_idx in range(2, 5):
        ws.column_dimensions[get_column_letter(col_idx)].width = 14
    ws.freeze_panes = 'B3'

    print(f"  Sheet {asset}详情 完成")

# ═══════════════════════════════════════════════════════════════════════════════
# Sheet 8: 月度收益率明细
# ═══════════════════════════════════════════════════════════════════════════════
ws8 = wb_out.create_sheet('月度收益明细')

ws8.merge_cells('A1:C1')
ws8['A1'] = '月度累计收益率明细（各标的融合策略）'
ws8['A1'].font = Font(bold=True, name='Arial', size=12, color='1F4E79')
ws8['A1'].alignment = CENTER
ws8['A1'].fill = PatternFill('solid', start_color='D6E4F0')

headers8 = ['年月', 'BTC融合%', 'ETH融合%', 'SOL融合%', 'DOGE融合%', '全组合%']
for col_idx, h in enumerate(headers8, 1):
    c = ws8.cell(row=2, column=col_idx, value=h)
    c.fill = HDR_FILL
    c.font = HDR_FONT
    c.alignment = CENTER
    c.border = BORDER

merged_maps = {asset: {dd: vv for dd, vv in merged_equity[asset]} for asset in ASSETS}
portfolio_map = {dd: vv for dd, vv in portfolio_equity}

for row_idx, d in enumerate(monthly_dates, 3):
    ws8.cell(row=row_idx, column=1, value=f'{d.year}-{d.month:02d}')
    ws8.cell(row=row_idx, column=1).alignment = CENTER
    ws8.cell(row=row_idx, column=1).border = BORDER
    ws8.cell(row=row_idx, column=1).font = BODY_FONT
    for col_idx, asset in enumerate(ASSETS, 2):
        v = get_val_at(merged_maps[asset], d)
        c = ws8.cell(row=row_idx, column=col_idx, value=round(v, 2))
        c.number_format = '0.00"%"'
        c.alignment = CENTER
        c.border = BORDER
        c.font = BODY_FONT
        if row_idx % 2 == 0:
            c.fill = ALT_FILL
        # 颜色：正收益绿，负收益红
        if v > 0:
            c.font = Font(name='Arial', size=10, color='375623')
        elif v < 0:
            c.font = Font(name='Arial', size=10, color='9C0006')
    v = get_val_at(portfolio_map, d)
    c = ws8.cell(row=row_idx, column=6, value=round(v, 2))
    c.number_format = '0.00"%"'
    c.alignment = CENTER
    c.border = BORDER
    c.font = Font(bold=True, name='Arial', size=10,
                  color='375623' if v > 0 else ('9C0006' if v < 0 else '000000'))
    if row_idx % 2 == 0:
        c.fill = ALT_FILL

for col_idx, w in enumerate([10, 12, 12, 12, 12, 12], 1):
    ws8.column_dimensions[get_column_letter(col_idx)].width = w
ws8.freeze_panes = 'A3'

print("  Sheet8 月度收益明细 完成")

# ─── 保存 ─────────────────────────────────────────────────────────────────────
out_path = '/Users/huminghe/Downloads/策略综合分析报告_2026-05-18.xlsx'
wb_out.save(out_path)
print(f"\n报告已保存: {out_path}")
print(f"包含 {len(wb_out.sheetnames)} 个sheet: {wb_out.sheetnames}")


# ─── 供 run_analysis.py 调用的接口 ────────────────────────────────────────────

def run(files: dict, base: str, out_dir: str, date_str: str = None):
    """
    files: {key: 绝对路径}，key 格式 ASSET_strat
    base:  路径前缀（run_analysis 传空字符串）
    """
    import datetime as _dt, os
    if date_str is None:
        date_str = _dt.date.today().isoformat()

    # 重建 ASSETS / strats_per_asset
    _assets = sorted(set(k.split("_")[0] for k in files))
    _strats_per: dict[str, list[str]] = {}
    for k in files:
        a, s = k.split("_", 1)
        _strats_per.setdefault(a, []).append(s)

    _all_stats: dict = {}
    _all_equity: dict = {}
    for k, fp in files.items():
        a, s = k.split("_", 1)
        _all_stats.setdefault(a, {})[s] = read_stats(base + fp)
        _all_equity.setdefault(a, {})[s] = read_equity_curve(base + fp)

    # 统一时间轴
    _all_dates_set = set()
    for a in _assets:
        for s in _strats_per.get(a, []):
            for dt, _ in _all_equity[a][s]:
                _all_dates_set.add(dt)
    _all_dates = sorted(_all_dates_set)

    # 插值
    def _interp(pts, dates):
        j = 0
        result = []
        for d in dates:
            while j < len(pts) - 1 and pts[j+1][0] <= d:
                j += 1
            result.append(pts[j][1] if pts[j][0] <= d else 0.0)
        return result

    _series = {}
    for k, fp in files.items():
        a, s = k.split("_", 1)
        pts = [(p[0], p[1]) for p in _all_equity[a][s]]
        _series[k] = _interp(pts, _all_dates)

    # 各标的融合
    _merged = {}
    for a in _assets:
        strats = _strats_per.get(a, [])
        keys = [f"{a}_{s}" for s in strats]
        _merged[a] = [sum(_series[k][i] for k in keys) / len(keys)
                      for i in range(len(_all_dates))]

    # 全组合
    _portfolio = [sum(_merged[a][i] for a in _assets) / len(_assets)
                  for i in range(len(_all_dates))]

    # 月度采样
    _monthly_dates = []
    _prev_ym = None
    for d in _all_dates:
        ym = (d.year, d.month)
        if ym != _prev_ym:
            _monthly_dates.append(d)
            _prev_ym = ym

    def _get_val_at(dm, d):
        best = None
        for dd, v in sorted(dm.items()):
            if dd <= d:
                best = v
        return best or 0.0

    # 构建 Excel
    from openpyxl import Workbook as _WB
    from openpyxl.utils import get_column_letter as _gcl
    import datetime as _dt2

    _wb = _WB()
    _wb.remove(_wb.active)

    # Sheet1: 总览对比
    _ws1 = _wb.create_sheet("总览对比")
    _headers = ["指标"] + [f"{a}_{s}" for a in _assets for s in _strats_per.get(a, [])]
    for ci, h in enumerate(_headers, 1):
        c = _ws1.cell(row=1, column=ci, value=h)
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = CENTER; c.border = BORDER
    for mi, (label, key, field, fmt, hb) in enumerate(METRICS):
        row = mi + 2
        _ws1.cell(row=row, column=1, value=label).font = BOLD_FONT
        ci = 2
        vals = {}
        for a in _assets:
            for s in _strats_per.get(a, []):
                v = get_key(_all_stats.get(a, {}).get(s, {}), key, field)
                if v is not None:
                    vals[(a, s)] = float(v)
        best_v = (max(vals.values()) if hb else min(vals.values())) if vals and hb is not None else None
        for a in _assets:
            for s in _strats_per.get(a, []):
                v = vals.get((a, s))
                c = _ws1.cell(row=row, column=ci)
                if v is not None:
                    c.value = v
                    c.number_format = '0.00"%"' if '%' in fmt else '0.000'
                    if best_v is not None and abs(v - best_v) < 1e-9:
                        c.fill = BEST_FILL
                        c.font = Font(bold=True, name='Arial', size=10, color='375623')
                    else:
                        c.font = BODY_FONT
                else:
                    c.value = '-'; c.font = BODY_FONT
                c.alignment = CENTER; c.border = BORDER
                ci += 1
    _ws1.column_dimensions['A'].width = 22
    for ci in range(2, len(_headers) + 1):
        _ws1.column_dimensions[_gcl(ci)].width = 11
    _ws1.freeze_panes = 'B2'

    # Sheet2: 月度收益明细
    _ws2 = _wb.create_sheet("月度收益明细")
    _h2 = ["年月"] + [f"{a}融合%" for a in _assets] + ["全组合%"]
    for ci, h in enumerate(_h2, 1):
        c = _ws2.cell(row=1, column=ci, value=h)
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = CENTER; c.border = BORDER
    _merged_maps = {a: {d: v for d, v in zip(_all_dates, _merged[a])} for a in _assets}
    _port_map = {d: v for d, v in zip(_all_dates, _portfolio)}
    for ri, d in enumerate(_monthly_dates, 2):
        _ws2.cell(row=ri, column=1, value=f"{d.year}-{d.month:02d}")
        for ci, a in enumerate(_assets, 2):
            v = _get_val_at(_merged_maps[a], d)
            c = _ws2.cell(row=ri, column=ci, value=round(v, 2))
            c.number_format = '0.00"%"'; c.alignment = CENTER; c.border = BORDER
        v = _get_val_at(_port_map, d)
        c = _ws2.cell(row=ri, column=len(_assets) + 2, value=round(v, 2))
        c.number_format = '0.00"%"'; c.alignment = CENTER; c.border = BORDER
    _ws2.freeze_panes = 'A2'

    _out = os.path.join(out_dir, f"策略综合分析报告_{date_str}.xlsx")
    _wb.save(_out)
    print(f"  Excel 报告已保存: {_out}")

