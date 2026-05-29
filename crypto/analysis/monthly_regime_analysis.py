"""
月度制度分析（2026-05-27）

按月统计4个策略（BTC/ETH/SOL/DOGE_ema）的合计盈亏，对照当月BTC市场状态。

结论：
- 亏损月26个，盈利月51个（2020-2026）
- 月末相对MA200%区分度最大（亏损月+2.9% vs 盈利月+16.0%）
- CI月均值>55时盈利月比例仅20%（5个月中4个亏损）
- ADX月均值>30时盈利月比例81%
- 月度指标是事后统计，无法用于实时决策

用法：
  python monthly_regime_analysis.py
"""
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import ccxt
import openpyxl
from pathlib import Path

downloads = Path('/Users/huminghe/Downloads')
files = {
    'BTC_ema':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
    'ETH_ema':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
    'SOL_ema':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
    'DOGE_ema': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
}

# ─── 加载交易数据 ─────────────────────────────────────────────────────────────

def load_trades(fname):
    wb = openpyxl.load_workbook(downloads / fname, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {v: i for i, v in enumerate(rows[0]) if v}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        num = row[col_idx['交易 #']]
        typ = str(row[col_idx['类型']])
        dt  = row[col_idx['日期和时间']]
        pnl = row[col_idx['净损益 USDT']]
        if num is None or dt is None: continue
        if num not in by_num: by_num[num] = {}
        if '进场' in typ:
            by_num[num]['entry_dt'] = pd.Timestamp(dt)
        elif '出场' in typ and pnl is not None:
            by_num[num]['exit_dt'] = pd.Timestamp(dt)
            by_num[num]['pnl'] = float(pnl)
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'exit_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    df['entry_dt'] = df['entry_dt'].dt.tz_localize(None)
    df['exit_dt']  = df['exit_dt'].dt.tz_localize(None)
    return df

# ─── 获取BTC日线市场数据 ──────────────────────────────────────────────────────

def get_btc_daily():
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601('2019-01-01T00:00:00Z')
    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
        if not batch: break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break

    df = pd.DataFrame(all_ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').sort_index()

    c = df['close']
    h = df['high']
    l = df['low']
    ret = c.pct_change()

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX
    up = h - h.shift()
    dn = l.shift() - l
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr14
    ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    # Choppiness Index(14)
    atr_sum  = tr.rolling(14).sum()
    hh = h.rolling(14).max()
    ll = l.rolling(14).min()
    ci = 100 * np.log10(atr_sum / (hh - ll).replace(0, np.nan)) / np.log10(14)

    # BBW
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bbw   = (std20 * 4) / ma20 * 100

    # MA200
    ma200 = c.rolling(200).mean()
    vs_ma200 = (c - ma200) / ma200 * 100

    # 月度收益率
    monthly_ret = c.resample('ME').last().pct_change() * 100

    result = pd.DataFrame({
        'close': c,
        'adx': adx,
        'ci': ci,
        'bbw': bbw,
        'vs_ma200': vs_ma200,
        'ret': ret * 100,
    })
    return result, monthly_ret

# ─── 按月聚合策略PnL ──────────────────────────────────────────────────────────

def monthly_pnl(trades_dict):
    """
    用出场时间归属月份（交易在哪个月结束，就算哪个月的盈亏）
    """
    all_rows = []
    for name, df in trades_dict.items():
        df = df.copy()
        df['strategy'] = name
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    combined['month'] = combined['exit_dt'].dt.to_period('M')

    monthly = combined.groupby('month').agg(
        total_pnl=('pnl', 'sum'),
        n_trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
        mean_pnl=('pnl', 'mean'),
    ).reset_index()
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()

    # 各策略单独月度PnL
    by_strat = combined.groupby(['month', 'strategy'])['pnl'].sum().unstack(fill_value=0)
    by_strat.index = by_strat.index.to_timestamp()

    return monthly, by_strat

# ─── 月度市场指标 ─────────────────────────────────────────────────────────────

def monthly_market(btc_daily):
    """每月取月末的市场指标值，以及月内均值"""
    m = btc_daily.resample('ME').agg({
        'adx':      'mean',   # 月内ADX均值
        'ci':       'mean',   # 月内CI均值
        'bbw':      'mean',   # 月内BBW均值
        'vs_ma200': 'last',   # 月末相对MA200
        'ret':      'sum',    # 月内日收益率之和（近似月收益率）
    })
    m.index = m.index.to_period('M').to_timestamp()
    return m

# ─── 打印月度详情表 ───────────────────────────────────────────────────────────

def print_monthly_table(monthly, by_strat, mkt):
    merged = monthly.set_index('month_dt').join(mkt, how='inner').join(by_strat, how='left')
    merged = merged.sort_index()

    # 只看有足够交易的月份
    merged = merged[merged['n_trades'] >= 5]

    w = [9, 8, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10]
    header = ['月份', 'BTC月收', 'ADX', 'CI', 'BBW', 'MA200%',
              '总PnL', '胜率', 'BTC_ema', 'ETH_ema', 'SOL_ema', 'DOGE_ema']

    print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
    print('│' + '│'.join(h.center(wi) for h, wi in zip(header, w)) + '│')
    print('├' + '┼'.join('─'*wi for wi in w) + '┤')

    for dt, row in merged.iterrows():
        month_str = dt.strftime('%Y-%m')
        btc_ret   = row.get('ret', float('nan'))
        adx_val   = row.get('adx', float('nan'))
        ci_val    = row.get('ci', float('nan'))
        bbw_val   = row.get('bbw', float('nan'))
        ma200_val = row.get('vs_ma200', float('nan'))
        total_pnl = row.get('total_pnl', float('nan'))
        wr        = row.get('win_rate', float('nan'))

        # 各策略PnL
        btc_p  = row.get('BTC_ema', float('nan'))
        eth_p  = row.get('ETH_ema', float('nan'))
        sol_p  = row.get('SOL_ema', float('nan'))
        doge_p = row.get('DOGE_ema', float('nan'))

        # 标记亏损月（总PnL为负）
        marker = ' ←' if total_pnl < 0 else ''

        cells = [
            month_str.center(w[0]),
            f'{btc_ret:+.1f}%'.center(w[1]) if not np.isnan(btc_ret) else '-'.center(w[1]),
            f'{adx_val:.1f}'.center(w[2])    if not np.isnan(adx_val) else '-'.center(w[2]),
            f'{ci_val:.1f}'.center(w[3])     if not np.isnan(ci_val)  else '-'.center(w[3]),
            f'{bbw_val:.1f}'.center(w[4])    if not np.isnan(bbw_val) else '-'.center(w[4]),
            f'{ma200_val:+.0f}%'.center(w[5]) if not np.isnan(ma200_val) else '-'.center(w[5]),
            f'{total_pnl:+.0f}{marker}'.center(w[6]),
            f'{wr:.0f}%'.center(w[7]),
            f'{btc_p:+.0f}'.center(w[8])   if not np.isnan(btc_p)  else '-'.center(w[8]),
            f'{eth_p:+.0f}'.center(w[9])   if not np.isnan(eth_p)  else '-'.center(w[9]),
            f'{sol_p:+.0f}'.center(w[10])  if not np.isnan(sol_p)  else '-'.center(w[10]),
            f'{doge_p:+.0f}'.center(w[11]) if not np.isnan(doge_p) else '-'.center(w[11]),
        ]
        print('│' + '│'.join(cells) + '│')

    print('└' + '┴'.join('─'*wi for wi in w) + '┘')
    return merged

# ─── 亏损月 vs 盈利月 市场指标对比 ───────────────────────────────────────────

def print_loss_vs_profit(merged):
    loss_months   = merged[merged['total_pnl'] < 0]
    profit_months = merged[merged['total_pnl'] >= 0]

    print(f"\n  亏损月: {len(loss_months)} 个  盈利月: {len(profit_months)} 个")

    indicators = [
        ('ret',      'BTC月收益%'),
        ('adx',      'ADX均值'),
        ('ci',       'CI均值'),
        ('bbw',      'BBW均值'),
        ('vs_ma200', '月末MA200%'),
        ('win_rate', '月胜率%'),
    ]

    w = [14, 10, 10, 10]
    print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
    header = ['指标', '亏损月均值', '盈利月均值', '差值']
    print('│' + '│'.join(h.center(wi) for h, wi in zip(header, w)) + '│')
    print('├' + '┼'.join('─'*wi for wi in w) + '┤')

    for col, label in indicators:
        lv = loss_months[col].dropna()
        pv = profit_months[col].dropna()
        if len(lv) == 0 or len(pv) == 0:
            continue
        lm, pm = lv.mean(), pv.mean()
        diff = lm - pm
        row = [
            label.ljust(w[0]-1),
            f'{lm:+.2f}'.center(w[1]),
            f'{pm:+.2f}'.center(w[2]),
            f'{diff:+.2f}'.center(w[3]),
        ]
        print('│' + '│'.join(row) + '│')

    print('└' + '┴'.join('─'*wi for wi in w) + '┘')
    print("  差值 = 亏损月均值 - 盈利月均值，负值=亏损月时该指标更低")

# ─── 主流程 ───────────────────────────────────────────────────────────────────

print("拉取BTC日线数据...")
btc_daily, _ = get_btc_daily()

print("加载交易数据...")
trades_dict = {}
for name, (fname, symbol) in files.items():
    trades_dict[name] = load_trades(fname)
    print(f"  {name}: {len(trades_dict[name])} 笔")

monthly, by_strat = monthly_pnl(trades_dict)
mkt = monthly_market(btc_daily)

print(f"\n{'='*100}")
print("  月度策略盈亏 × 市场状态（按出场时间归月，← 标记亏损月）")
print('='*100)
print("  列说明：BTC月收=BTC当月涨跌幅  ADX=月内ADX均值  CI=月内震荡指数均值")
print("          BBW=布林带宽均值  MA200%=月末价格相对MA200偏离  总PnL=4策略合计")
print()

merged = print_monthly_table(monthly, by_strat, mkt)

print(f"\n{'='*60}")
print("  亏损月 vs 盈利月 市场指标对比")
print('='*60)
print_loss_vs_profit(merged)

# ─── CI分箱：月度CI均值与策略表现 ────────────────────────────────────────────

print(f"\n{'='*60}")
print("  月度CI均值分箱 vs 策略总PnL")
print('='*60)

bins   = [0, 45, 50, 55, 60, 65, 100]
labels = ['<45(强趋势)', '45~50', '50~55', '55~60', '60~65', '>65(强横盘)']
merged['ci_bin'] = pd.cut(merged['ci'], bins=bins, labels=labels)

w = [14, 7, 10, 8]
print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
print('│' + '│'.join(h.center(wi) for h, wi in zip(['CI区间','月数','均值PnL','盈利月%'], w)) + '│')
print('├' + '┼'.join('─'*wi for wi in w) + '┤')
for label in labels:
    sub = merged[merged['ci_bin'] == label]
    if len(sub) == 0: continue
    n    = len(sub)
    mpnl = sub['total_pnl'].mean()
    wp   = (sub['total_pnl'] >= 0).mean() * 100
    row  = [label.center(w[0]), str(n).center(w[1]),
            f'{mpnl:+.0f}'.center(w[2]), f'{wp:.0f}%'.center(w[3])]
    print('│' + '│'.join(row) + '│')
print('└' + '┴'.join('─'*wi for wi in w) + '┘')

# ─── ADX分箱 ─────────────────────────────────────────────────────────────────

print(f"\n  月度ADX均值分箱 vs 策略总PnL")
bins2   = [0, 15, 20, 25, 30, 100]
labels2 = ['<15(极弱)', '15~20(弱)', '20~25(中)', '25~30(强)', '>30(极强)']
merged['adx_bin'] = pd.cut(merged['adx'], bins=bins2, labels=labels2)

print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
print('│' + '│'.join(h.center(wi) for h, wi in zip(['ADX区间','月数','均值PnL','盈利月%'], w)) + '│')
print('├' + '┼'.join('─'*wi for wi in w) + '┤')
for label in labels2:
    sub = merged[merged['adx_bin'] == label]
    if len(sub) == 0: continue
    n    = len(sub)
    mpnl = sub['total_pnl'].mean()
    wp   = (sub['total_pnl'] >= 0).mean() * 100
    row  = [label.center(w[0]), str(n).center(w[1]),
            f'{mpnl:+.0f}'.center(w[2]), f'{wp:.0f}%'.center(w[3])]
    print('│' + '│'.join(row) + '│')
print('└' + '┴'.join('─'*wi for wi in w) + '┘')
