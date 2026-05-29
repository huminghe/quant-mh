"""
制度信号 × 实盘PnL 验证框架（2026-05-27）

用实盘交易数据验证制度信号的有效性。
读取 Downloads 中的 xlsx 交易清单，匹配入场时的市场状态，
对比 risk_on/risk_off 下的实际PnL分布。

关键发现：双向策略在大跌时做空盈利，BTC价格下跌≠策略亏损。
          横盘（低ADX+低ATR）才是策略真正的敌人。

用法：
  python regime_pnl_validate.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import ccxt
import openpyxl
from pathlib import Path

# ─── 1. 读取所有策略交易清单 ──────────────────────────────────────────────────
downloads = Path('/Users/huminghe/Downloads')
# 取最新版本的文件（按日期排序取最新）
files = {
    'BTC_ema': 'strategy_ema_btc_OKX_BTCUSDT.P_2026-05-22_19c0f.xlsx',
    'ETH_ema': 'strategy_ema_eth_OKX_ETHUSDT.P_2026-05-22_3004a.xlsx',
    'SOL_ema': 'strategy_ema_sol_OKX_SOLUSDT.P_2026-05-22_9ec54.xlsx',
    'DOGE_ema': 'strategy_ema_meme_OKX_DOGEUSDT.P_2026-05-22_28c99.xlsx',
}

all_trades = []
for strat, fname in files.items():
    path = downloads / fname
    if not path.exists():
        print(f"文件不存在：{fname}")
        continue
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['交易清单']
    rows = list(ws.iter_rows(values_only=True))
    # 找表头
    header = rows[0]
    col_idx = {v: i for i, v in enumerate(header) if v}
    data = []
    for row in rows[1:]:
        if row[0] is None: continue
        try:
            trade_num = row[col_idx['交易 #']]
            trade_type = row[col_idx['类型']]
            dt = row[col_idx['日期和时间']]
            pnl = row[col_idx['净损益 USDT']]
            if trade_num and dt and pnl is not None:
                # 只取出场记录（有完整PnL）
                if '出场' in str(trade_type):
                    data.append({'strategy': strat, 'date': pd.Timestamp(dt), 'pnl': float(pnl)})
        except: continue
    wb.close()
    df_t = pd.DataFrame(data)
    all_trades.append(df_t)
    print(f"{strat}: {len(df_t)} 笔已平仓交易，{df_t['date'].min().date()} ~ {df_t['date'].max().date()}")

trades = pd.concat(all_trades, ignore_index=True)
trades['date'] = trades['date'].dt.tz_localize(None).dt.normalize()
print(f"\n合计 {len(trades)} 笔交易")

# ─── 2. 拉 BTC 日线，计算制度信号 ────────────────────────────────────────────
print("\n拉取 BTC 日线数据...")
exchange = ccxt.binance({'enableRateLimit': True})
since = exchange.parse8601('2019-01-01T00:00:00Z')
all_ohlcv = []
while True:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
    if not ohlcv: break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 86400000
    if len(ohlcv) < 1000: break

df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.normalize().dt.tz_localize(None)
df = df.set_index('date').drop(columns=['timestamp'])
df = df[~df.index.duplicated()].sort_index()
returns = df['close'].pct_change().fillna(0)

# 计算各信号
ma200  = df['close'].rolling(200).mean()
ret_21 = df['close'].pct_change(21)
ret_30 = df['close'].pct_change(30)
vol_20 = returns.rolling(20).std()
vol_z  = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period,adjust=False).mean()
    up, down = high.diff(), -low.diff()
    dm_p = up.where((up>down)&(up>0),0)
    dm_m = down.where((down>up)&(down>0),0)
    di_p = 100*dm_p.ewm(alpha=1/period,adjust=False).mean()/atr
    di_m = 100*dm_m.ewm(alpha=1/period,adjust=False).mean()/atr
    dx = (100*(di_p-di_m).abs()/(di_p+di_m)).fillna(0)
    return dx.ewm(alpha=1/period,adjust=False).mean()

adx = compute_adx(df)
atr_raw = pd.concat([df['high']-df['low'],(df['high']-df['close'].shift()).abs(),(df['low']-df['close'].shift()).abs()],axis=1).max(axis=1).rolling(14).mean()
atr_pct = atr_raw.rolling(252).rank(pct=True)

# 制度信号（之前推荐方案：MA200+跌幅10%+30日动量，>=3/3）
s_ma200 = (df['close'] < ma200).fillna(False)
s_drop  = (ret_21 < -0.10).fillna(False)
s_ret30 = (ret_30 < 0).fillna(False)
score3  = s_ma200.astype(int) + s_drop.astype(int) + s_ret30.astype(int)
regime_recommended = (score3 >= 3).astype(int)

# 横盘制度（ADX低+波动率低）：策略真正的敌人
s_adx_low = (adx < 20).fillna(False)
s_vol_low = (atr_pct < 0.40).fillna(False)  # 低波动
regime_chop = (s_adx_low.astype(int) + s_vol_low.astype(int) >= 2).astype(int)

# 合并到交易数据
trades['regime_rec']  = trades['date'].map(regime_recommended)
trades['regime_chop'] = trades['date'].map(regime_chop)
trades['adx']         = trades['date'].map(adx)
trades['atr_pct']     = trades['date'].map(atr_pct)
trades['btc_ret21']   = trades['date'].map(ret_21)
trades['btc_above_ma'] = trades['date'].map(df['close'] > ma200)

# ─── 3. 核心分析：各制度下策略表现 ───────────────────────────────────────────
print("\n=== 推荐方案（MA200+跌幅10%+30日动量，>=3/3）下各策略表现 ===")
print(f"{'策略':<12} {'risk_off笔数':>12} {'risk_off均PnL':>14} {'risk_on笔数':>12} {'risk_on均PnL':>13} {'区分度':>10}")
print("-" * 70)
for strat in files.keys():
    t = trades[trades['strategy'] == strat].dropna(subset=['regime_rec'])
    off = t[t['regime_rec'] == 1]
    on  = t[t['regime_rec'] == 0]
    if len(off) < 3: continue
    print(f"  {strat:<10} {len(off):>12} {off['pnl'].mean():>+13.1f}  {len(on):>12} {on['pnl'].mean():>+12.1f}  {off['pnl'].mean()-on['pnl'].mean():>+9.1f}")

print("\n=== 横盘制度（ADX<20 + ATR百分位<40%）下各策略表现 ===")
print(f"{'策略':<12} {'横盘笔数':>10} {'横盘均PnL':>12} {'非横盘笔数':>12} {'非横盘均PnL':>13} {'区分度':>10}")
print("-" * 70)
for strat in files.keys():
    t = trades[trades['strategy'] == strat].dropna(subset=['regime_chop'])
    chop = t[t['regime_chop'] == 1]
    trend = t[t['regime_chop'] == 0]
    if len(chop) < 3: continue
    print(f"  {strat:<10} {len(chop):>10} {chop['pnl'].mean():>+11.1f}  {len(trend):>12} {trend['pnl'].mean():>+12.1f}  {chop['pnl'].mean()-trend['pnl'].mean():>+9.1f}")

# ─── 4. ADX 分桶：策略实际 PnL ────────────────────────────────────────────────
print("\n=== ADX 分桶：策略实际每笔 PnL 均值（合并4个标的）===")
print(f"{'ADX区间':<15} {'笔数':>6} {'均PnL':>10} {'胜率':>8}")
print("-" * 45)
bins = [(0,15,'极弱'),(15,20,'弱'),(20,25,'中等'),(25,35,'强'),(35,100,'极强')]
for lo, hi, label in bins:
    mask = (trades['adx'] >= lo) & (trades['adx'] < hi)
    t = trades[mask]
    if len(t) < 5: continue
    win_rate = (t['pnl'] > 0).mean() * 100
    print(f"  ADX {lo:>2}~{hi:<3} {label:<4}  {len(t):>6}  {t['pnl'].mean():>+9.1f}  {win_rate:>7.1f}%")

# ─── 5. ATR百分位 分桶 ────────────────────────────────────────────────────────
print("\n=== ATR百分位 分桶：策略实际每笔 PnL 均值 ===")
print(f"{'ATR百分位':<15} {'笔数':>6} {'均PnL':>10} {'胜率':>8}")
print("-" * 45)
bins_atr = [(0,0.25,'低波动'),(0.25,0.50,'中低'),(0.50,0.75,'中高'),(0.75,1.0,'高波动')]
for lo, hi, label in bins_atr:
    mask = (trades['atr_pct'] >= lo) & (trades['atr_pct'] < hi)
    t = trades[mask]
    if len(t) < 5: continue
    win_rate = (t['pnl'] > 0).mean() * 100
    print(f"  ATR {lo:.2f}~{hi:.2f} {label:<4}  {len(t):>6}  {t['pnl'].mean():>+9.1f}  {win_rate:>7.1f}%")

# ─── 6. 结论：什么制度对策略最有害 ───────────────────────────────────────────
print("\n=== 四象限分析：MA200位置 × ADX强弱 ===")
print(f"{'制度':<25} {'笔数':>6} {'均PnL':>10} {'胜率':>8} {'说明'}")
print("-" * 65)
quadrants = [
    (trades['btc_above_ma'] & (trades['adx']<20),  '牛市横盘（MA上+ADX低）', '趋势策略最差？'),
    (trades['btc_above_ma'] & (trades['adx']>=20), '牛市趋势（MA上+ADX高）', '应该最好'),
    (~trades['btc_above_ma'] & (trades['adx']<20), '熊市横盘（MA下+ADX低）', '双向策略如何？'),
    (~trades['btc_above_ma'] & (trades['adx']>=20),'熊市趋势（MA下+ADX高）', '做空盈利？'),
]
for mask, label, note in quadrants:
    t = trades[mask.reindex(trades.index, fill_value=False)]
    if len(t) < 5: continue
    win_rate = (t['pnl'] > 0).mean() * 100
    print(f"  {label:<23} {len(t):>6}  {t['pnl'].mean():>+9.1f}  {win_rate:>7.1f}%  {note}")

