"""
# ARCHIVED: 结论已固化到 docs/strategy_research_log.md 或 docs/filters_validation.md，不再需要运行
EMA 差值 ATR 阈值过滤验证（2026-06-01）【已完结，结论：无效】

结论：
  ATR 阈值过滤对差值变化方向策略无效。k 越大 Sharpe 越低，净利润越低。
  v1 基准 Sharpe 1.571，k=0.05 降至 1.444，k=0.30 降至 1.092。
  v2 基准 Sharpe 2.477，k=0.05 降至 2.283，k=0.30 降至 1.992。
  根本原因：趋势启动时差值小，过滤掉的恰好是最好的入场点，与入场过滤器失效原因相同。
  结论已写入 strategy_research_log.md 和 lessons.md，此脚本归档备查。

背景：
  当前策略用双 EMA 差值变化方向作为信号（差值上升持仓，下降平仓）。
  目标：验证加入 "差值/价格 > k × ATR/价格" 过滤条件后，
  过滤掉不满足条件的交易，是否能提升 Sharpe 和净利润。

方法：
  1. 从 TV 导出的交易清单中读取每笔交易的入场时间和 PnL
  2. 从 Binance API 拉取 8H OHLCV 数据
  3. 对每笔交易入场时刻，计算 EMA7、EMA25 差值和 ATR(20)
  4. 网格搜索 k ∈ [0.05, 0.10, ..., 0.60]，过滤不满足条件的交易
  5. 对比各 k 值下的 Sharpe、净利润、胜率、交易数

用法：
  python ema_diff_atr_filter.py
  python ema_diff_atr_filter.py --version v1   # 只跑 v1
  python ema_diff_atr_filter.py --version v2   # 只跑 v2
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, numpy as np, pandas as pd, ccxt, openpyxl
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--version', choices=['v1', 'v2', 'both'], default='both')
args = parser.parse_args()

EMA_FAST   = 7
EMA_SLOW   = 25
ATR_PERIOD = 20
K_VALUES   = np.arange(0.05, 0.65, 0.05)   # 0.05 ~ 0.60，共 12 个
N_YEARS    = 7                               # 回测年数，用于 Sharpe 年化

downloads = Path('/Users/huminghe/Downloads')

VERSION_FILES = {
    'v1': {
        'BTC':  ('strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',  'BTC/USDT'),
        'ETH':  ('strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',   'ETH/USDT'),
        'SOL':  ('strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',   'SOL/USDT'),
        'DOGE': ('strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx', 'DOGE/USDT'),
    },
    'v2': {
        'BTC':  ('v2_strategy_btc_OKX_BTCUSDT.P_2026-05-22_c2fde.xlsx',   'BTC/USDT'),
        'ETH':  ('v2_strategy_eth_OKX_ETHUSDT.P_2026-05-22_0f7c1.xlsx',   'ETH/USDT'),
        'SOL':  ('v2_strategy_sol_OKX_SOLUSDT.P_2026-05-22_6d6ed.xlsx',   'SOL/USDT'),
        'DOGE': ('v2_strategy_doge_OKX_DOGEUSDT.P_2026-05-22_8856b.xlsx', 'DOGE/USDT'),
    },
}

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_trades(fname):
    """加载交易清单，TV UTC+8 时间 -8h 转 UTC"""
    path = downloads / fname
    if not path.exists():
        print(f'  文件不存在：{fname}')
        return pd.DataFrame()
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = '交易清单' if '交易清单' in wb.sheetnames else 'Trades'
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    col_idx = {name: i for i, name in enumerate(rows[0]) if name is not None}
    by_num = {}
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            num = row[col_idx.get('交易 #', col_idx.get('Trade #', -1))]
            typ = str(row[col_idx.get('类型', col_idx.get('Type', -1))])
            dt  = row[col_idx.get('日期和时间', col_idx.get('Date/Time', -1))]
            pnl_key = '净损益 USDT' if '净损益 USDT' in col_idx else 'Profit USDT'
            pnl = row[col_idx.get(pnl_key, -1)]
            price_key = '价格 USDT' if '价格 USDT' in col_idx else 'Price USDT'
            price = row[col_idx.get(price_key, -1)]
            if num is None or dt is None: continue
            if num not in by_num: by_num[num] = {}
            if '进场' in typ or 'Entry' in typ:
                by_num[num]['entry_dt'] = pd.Timestamp(dt)
                if price is not None:
                    by_num[num]['entry_price'] = float(price)
            elif ('出场' in typ or 'Exit' in typ) and pnl is not None:
                by_num[num]['pnl'] = float(pnl)
        except: continue
    wb.close()
    rows_out = [d for d in by_num.values() if 'entry_dt' in d and 'pnl' in d]
    df = pd.DataFrame(rows_out)
    # TV 导出时间是 UTC+8，减 8 小时转为 UTC
    df['entry_dt'] = (pd.to_datetime(df['entry_dt']) - pd.Timedelta(hours=8)).astype('datetime64[ns]')
    return df.sort_values('entry_dt').reset_index(drop=True)

ohlcv_cache = {}
def fetch_ohlcv(symbol):
    if symbol in ohlcv_cache: return ohlcv_cache[symbol]
    print(f'  拉取 {symbol} 8H...', end=' ', flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    all_bars, since = [], ex.parse8601('2019-01-01T00:00:00Z')
    while True:
        for attempt in range(3):
            try:
                bars = ex.fetch_ohlcv(symbol, '8h', since=since, limit=1000)
                break
            except Exception as e:
                if attempt == 2: raise
                import time; time.sleep(3)
        if not bars: break
        all_bars.extend(bars)
        if len(bars) < 1000: break
        since = bars[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms').astype('datetime64[ns]')
    df = df.set_index('dt').sort_index()
    ohlcv_cache[symbol] = df
    print('done')
    return df

# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def calc_indicators(ohlcv):
    """计算 EMA7、EMA25 差值（/价格标准化）和 ATR(20)/价格"""
    df = ohlcv.copy()
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['diff_pct'] = (df['ema_fast'] - df['ema_slow']) / df['close']   # 差值/价格

    # ATR(20) Wilder 平滑
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr_pct'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean() / df['close']  # ATR/价格

    return df[['diff_pct', 'atr_pct']]

# ─── 统计 ─────────────────────────────────────────────────────────────────────

def calc_stats(pnl_series):
    """计算 Sharpe、净利润、胜率、盈亏比"""
    if len(pnl_series) == 0:
        return dict(n=0, total=0, win_rate=0, rr=0, sharpe=0)
    total    = pnl_series.sum()
    win_rate = (pnl_series > 0).mean() * 100
    wins     = pnl_series[pnl_series > 0]
    losses   = pnl_series[pnl_series < 0]
    rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
    # Sharpe：年化，假设每年 N_YEARS 年总交易数均匀分布
    n = len(pnl_series)
    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(n / N_YEARS) if pnl_series.std() > 0 else 0
    return dict(n=n, total=round(total,1), win_rate=round(win_rate,1), rr=round(rr,2), sharpe=round(sharpe,3))

# ─── 主流程 ───────────────────────────────────────────────────────────────────

def run_version(version):
    print(f'\n{"="*60}')
    print(f'版本：{version.upper()}')
    print(f'{"="*60}')

    # 收集所有标的的交易记录 + 指标
    all_trades = []
    for asset, (fname, symbol) in VERSION_FILES[version].items():
        print(f'\n[{asset}]')
        trades = load_trades(fname)
        if trades.empty:
            print(f'  跳过（无交易记录）')
            continue
        ohlcv = fetch_ohlcv(symbol)
        indicators = calc_indicators(ohlcv)

        # merge_asof：用入场时刻匹配最近的 8H bar 指标
        ind_reset = indicators.reset_index()
        ind_reset.columns = ['dt', 'diff_pct', 'atr_pct']
        merged = pd.merge_asof(
            trades.sort_values('entry_dt'),
            ind_reset.sort_values('dt'),
            left_on='entry_dt', right_on='dt',
            direction='backward'
        )
        merged['asset'] = asset
        all_trades.append(merged)
        print(f'  交易数：{len(merged)}，diff_pct 均值：{merged["diff_pct"].mean():.4f}，atr_pct 均值：{merged["atr_pct"].mean():.4f}')

    if not all_trades:
        print('无有效数据')
        return

    df = pd.concat(all_trades, ignore_index=True)
    total_trades = len(df)
    print(f'\n合计交易数：{total_trades}')

    # 基准（不过滤）
    base = calc_stats(df['pnl'])
    print(f'\n{"k":>6} {"保留%":>7} {"交易数":>6} {"净利润":>10} {"胜率%":>7} {"盈亏比":>7} {"Sharpe":>8}')
    print(f'{"基准":>6} {"100%":>7} {base["n"]:>6} {base["total"]:>10} {base["win_rate"]:>7} {base["rr"]:>7} {base["sharpe"]:>8}')
    print('-' * 60)

    # 网格搜索
    results = []
    for k in K_VALUES:
        # 过滤条件：|diff_pct| > k × atr_pct（差值绝对值，因为多空双向）
        mask = df['diff_pct'].abs() > k * df['atr_pct']
        filtered = df[mask]['pnl']
        s = calc_stats(filtered)
        keep_pct = len(filtered) / total_trades * 100
        results.append({'k': k, 'keep_pct': keep_pct, **s})
        print(f'{k:>6.2f} {keep_pct:>6.1f}% {s["n"]:>6} {s["total"]:>10} {s["win_rate"]:>7} {s["rr"]:>7} {s["sharpe"]:>8}')

    # 找最优 k（Sharpe 最高）
    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df['sharpe'].idxmax()]
    print(f'\n最优 k = {best["k"]:.2f}，Sharpe {base["sharpe"]} → {best["sharpe"]}，保留 {best["keep_pct"]:.1f}% 交易')

versions = ['v1', 'v2'] if args.version == 'both' else [args.version]
for v in versions:
    run_version(v)
