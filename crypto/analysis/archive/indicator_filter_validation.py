"""
制度过滤指标验证（2026-05-27）

对比盈利 vs 亏损交易在入场时的8个市场指标，验证是否有预测能力。

指标清单：
1. ADX(14)          - 趋势强度
2. ATR百分位          - 波动率历史分位
3. BBW布林带宽         - 价格带宽（横盘程度）
4. 相对MA200%        - 价格相对长期均线位置
5. 21日动量%          - 近期价格动量
6. 年化波动率%         - 近期波动绝对水平
7. Choppiness Index - 价格运动效率（专门衡量趋势vs横盘）
8. 收益率自相关(lag5)   - 近期收益率是否有趋势延续性

结论：所有指标区分度接近0（最大差值2.0），无实用价值。
      低胜率+高盈亏比策略的亏损是随机分布的，不集中于特定市场环境。

用法：
  python indicator_filter_validation.py
"""
2. ATR百分位          - 波动率历史分位
3. BBW布林带宽         - 价格带宽（横盘程度）
4. 相对MA200%        - 价格相对长期均线位置
5. 21日动量%          - 近期价格动量
6. 年化波动率%         - 近期波动绝对水平
7. Choppiness Index - 价格运动效率（专门衡量趋势vs横盘）
8. 收益率自相关(lag5)   - 近期收益率是否有趋势延续性
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
    df['entry_dt'] = df['entry_dt'].dt.tz_localize(None).astype('datetime64[ms]')
    return df

# ─── 获取市场特征数据 ─────────────────────────────────────────────────────────

market_cache = {}

def get_market_features(symbol, timeframe='1d'):
    key = (symbol, timeframe)
    if key in market_cache:
        return market_cache[key]

    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601('2019-01-01T00:00:00Z')
    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch: break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000: break

    df = pd.DataFrame(all_ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[ms]')
    df = df.set_index('dt').sort_index()

    c = df['close']
    h = df['high']
    l = df['low']
    ret = c.pct_change()

    # 1. ADX(14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h - h.shift()
    dn = l.shift() - l
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr14
    ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    # 2. ATR百分位（252日滚动）
    atr_pct = atr14.rolling(252).rank(pct=True) * 100

    # 3. BBW布林带宽
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bbw = (std20 * 4) / ma20 * 100

    # 4. 相对MA200%
    ma200 = c.rolling(200).mean()
    price_vs_ma200 = (c - ma200) / ma200 * 100

    # 5. 21日动量%
    mom21 = c.pct_change(21) * 100

    # 6. 年化波动率%
    vol20 = ret.rolling(20).std() * np.sqrt(365) * 100

    # 7. Choppiness Index(14)
    # CI = 100 × log10(sum(ATR_1, N) / (highest_high_N - lowest_low_N)) / log10(N)
    # 值越高 = 越横盘(choppy)，值越低 = 越趋势
    # 阈值：< 38.2 强趋势，> 61.8 强横盘
    N_ci = 14
    atr_sum = tr.rolling(N_ci).sum()
    hh = h.rolling(N_ci).max()
    ll = l.rolling(N_ci).min()
    range_hl = (hh - ll).replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / range_hl) / np.log10(N_ci)

    # 8. 收益率自相关(lag5, 40日窗口)
    # 正值 = 趋势延续（今天涨，5天后也倾向于涨）
    # 负值 = 均值回归（今天涨，5天后倾向于跌）
    def rolling_autocorr(series, lag=5, window=40):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window + lag, len(series)):
            s = series.iloc[i-window:i]
            result.iloc[i] = s.autocorr(lag=lag)
        return result

    autocorr5 = rolling_autocorr(ret, lag=5, window=40)

    result = pd.DataFrame({
        'adx': adx,
        'atr_pct': atr_pct,
        'bbw': bbw,
        'price_vs_ma200': price_vs_ma200,
        'mom21': mom21,
        'vol20': vol20,
        'ci': ci,
        'autocorr5': autocorr5,
    })
    market_cache[key] = result
    return result

# ─── 合并交易与市场特征 ───────────────────────────────────────────────────────

def merge_features(trades, symbol, timeframe='1d'):
    mkt = get_market_features(symbol, timeframe).reset_index()
    mkt['dt'] = mkt['dt'].astype('datetime64[ms]')
    trades_sorted = trades.sort_values('entry_dt')
    merged = pd.merge_asof(
        trades_sorted,
        mkt.rename(columns={'dt': 'entry_dt'}),
        on='entry_dt',
        direction='backward'
    )
    return merged

# ─── 打印汇总表 ───────────────────────────────────────────────────────────────

feat_names = {
    'adx':            'ADX(14)',
    'atr_pct':        'ATR百分位',
    'bbw':            'BBW布林带宽',
    'price_vs_ma200': '相对MA200%',
    'mom21':          '21日动量%',
    'vol20':          '年化波动率%',
    'ci':             'Choppiness(14)',
    'autocorr5':      '收益自相关lag5',
}

def print_feature_table(name, df):
    win  = df[df['pnl'] > 0]
    lose = df[df['pnl'] <= 0]
    feats = list(feat_names.keys())

    w = [16, 10, 10, 10, 10, 10, 10]
    print(f"\n{'─'*78}")
    print(f"  {name}  盈利({len(win)}笔) vs 亏损({len(lose)}笔)  入场时日线指标对比")
    print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
    header = ['指标', '盈利均值', '亏损均值', '差值', '盈利中位', '亏损中位', '差值(中位)']
    print('│' + '│'.join(h.center(wi) for h, wi in zip(header, w)) + '│')
    print('├' + '┼'.join('─'*wi for wi in w) + '┤')

    discriminations = {}
    for feat in feats:
        wv = win[feat].dropna()
        lv = lose[feat].dropna()
        if len(wv) < 5 or len(lv) < 5:
            continue
        wm, lm = wv.mean(), lv.mean()
        wmed, lmed = wv.median(), lv.median()
        diff_mean = wm - lm
        diff_med  = wmed - lmed
        discriminations[feat] = diff_mean

        fname = feat_names[feat]
        row = [
            fname.ljust(w[0]-1),
            f'{wm:+.2f}'.center(w[1]),
            f'{lm:+.2f}'.center(w[2]),
            f'{diff_mean:+.2f}'.center(w[3]),
            f'{wmed:+.2f}'.center(w[4]),
            f'{lmed:+.2f}'.center(w[5]),
            f'{diff_med:+.2f}'.center(w[6]),
        ]
        print('│' + '│'.join(row) + '│')

    print('└' + '┴'.join('─'*wi for wi in w) + '┘')
    return discriminations

# ─── 主流程 ───────────────────────────────────────────────────────────────────

print("加载交易数据并拉取市场特征（首次需要几分钟）...")
all_disc = {}

for strat_name, (fname, symbol) in files.items():
    print(f"\n{'='*60}")
    print(f"  {strat_name}  ({symbol})")
    print('='*60)
    trades = load_trades(fname)
    print(f"  {len(trades)} 笔交易，计算日线指标...")
    df = merge_features(trades, symbol, '1d')
    disc = print_feature_table(strat_name, df)
    all_disc[strat_name] = disc

# ─── 汇总表 ───────────────────────────────────────────────────────────────────

print(f"\n{'='*78}")
print("  汇总：各指标区分度（盈利均值 - 亏损均值）")
print('='*78)

feats  = list(feat_names.keys())
strats = list(all_disc.keys())
w = [16] + [12]*len(strats) + [10]
header = ['指标'] + strats + ['平均']
print('┌' + '┬'.join('─'*wi for wi in w) + '┐')
print('│' + '│'.join(h.center(wi) for h, wi in zip(header, w)) + '│')
print('├' + '┼'.join('─'*wi for wi in w) + '┤')

for feat in feats:
    vals = [all_disc[s].get(feat, float('nan')) for s in strats]
    avg  = np.nanmean(vals)
    row  = [feat_names[feat].ljust(w[0]-1)]
    for v, wi in zip(vals, w[1:]):
        row.append(f'{v:+.2f}'.center(wi))
    row.append(f'{avg:+.2f}'.center(w[-1]))
    print('│' + '│'.join(row) + '│')

print('└' + '┴'.join('─'*wi for wi in w) + '┘')
print()
print("  指标说明：")
print("  ADX(14)         趋势强度，0~20横盘，20~35趋势，35+强趋势")
print("  ATR百分位         当前ATR在过去252日的排名（0~100），越高波动越大")
print("  BBW布林带宽        布林带宽度/中轨×100，越小越横盘")
print("  相对MA200%        (价格-MA200)/MA200×100，正=牛市，负=熊市")
print("  21日动量%         过去21日涨跌幅，正=近期上涨")
print("  年化波动率%        20日滚动标准差年化，绝对波动水平")
print("  Choppiness(14)  价格运动效率，<38.2强趋势，>61.8强横盘")
print("  收益自相关lag5     40日窗口lag-5自相关，正=趋势延续，负=均值回归")
print()
print("  正差值=盈利交易时该指标更高；负差值=亏损交易时该指标更高")
print("  差值绝对值越大，预测能力越强")
