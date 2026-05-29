"""
规则投票方案A 阈值优化分析（2026-05-27）

1. 各信号单独的区分度（MA200/ADX/ATR百分位/VolZ）
2. 不同阈值组合的效果
3. 资金费率真实数据 vs 波动率Z-score 替代

结论：规则投票方案在BTC价格层面有效，但对双向策略无意义（大跌=做空盈利）。
      真正有效的过滤是横盘检测（低ADX+低ATR），而非价格方向。

用法：
  python regime_rule_optimize.py
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt

# ─── 拉数据 ───────────────────────────────────────────────────────────────────
print("拉取数据...")
exchange = ccxt.binance({'enableRateLimit': True})
since = exchange.parse8601('2020-01-01T00:00:00Z')
all_ohlcv = []
while True:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
    if not ohlcv:
        break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 86400000
    if len(ohlcv) < 1000:
        break

df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.normalize()
df = df.set_index('date').drop(columns=['timestamp'])
df = df[~df.index.duplicated()].sort_index()
returns = df['close'].pct_change().fillna(0)

# ─── 计算各信号 ───────────────────────────────────────────────────────────────
def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    dm_plus = up.where((up > down) & (up > 0), 0)
    dm_minus = down.where((down > up) & (down > 0), 0)
    di_plus = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean()

ma200 = df['close'].rolling(200).mean()
adx = compute_adx(df, 14)
atr_raw = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1).rolling(14).mean()
atr_pct = atr_raw.rolling(252).rank(pct=True)
vol_20 = returns.rolling(20).std()
vol_zscore = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()

# 额外信号：30日收益率（动量）
ret_30 = df['close'].pct_change(30)
# 额外信号：价格跌幅（与暂停机制一致）
ret_21 = df['close'].pct_change(21)  # 3周跌幅

valid_idx = df.index[252:]  # 需要足够历史
r = returns[valid_idx]

# ─── 1. 各信号单独区分度 ──────────────────────────────────────────────────────
print("\n=== 各信号单独区分度 ===")
print(f"{'信号':<30} {'触发天数':>8} {'触发期日均收益':>14} {'未触发期日均收益':>16} {'区分度':>8}")
print("-" * 80)

signals = {
    'MA200（价格<MA200）':       (df['close'] < ma200)[valid_idx],
    'ADX<20（无趋势）':          (adx < 20)[valid_idx],
    'ADX<25（弱趋势）':          (adx < 25)[valid_idx],
    'ATR百分位>75%':             (atr_pct > 0.75)[valid_idx],
    'ATR百分位>65%':             (atr_pct > 0.65)[valid_idx],
    '波动率Z-score>1.5':         (vol_zscore > 1.5)[valid_idx],
    '波动率Z-score>2':           (vol_zscore > 2)[valid_idx],
    '21日跌幅<-10%':             (ret_21 < -0.10)[valid_idx],
    '21日跌幅<-12%':             (ret_21 < -0.12)[valid_idx],
    '30日收益<0':                (ret_30 < 0)[valid_idx],
}

for name, sig in signals.items():
    sig = sig.fillna(False)
    n_on = sig.sum()
    if n_on < 10:
        continue
    ret_on = r[sig].mean() * 100
    ret_off = r[~sig].mean() * 100
    diff = ret_on - ret_off
    print(f"  {name:<28} {n_on:>8} {ret_on:>+13.3f}% {ret_off:>+15.3f}% {diff:>+7.3f}%")

# ─── 2. 不同阈值组合测试 ──────────────────────────────────────────────────────
print("\n=== 不同触发阈值对比（>=N/总信号数 触发risk_off）===")
print(f"{'组合':<45} {'risk_off%':>9} {'off日均收益':>12} {'on日均收益':>12} {'翻转次数':>8} {'区分度':>8}")
print("-" * 100)

# 构建信号矩阵（4个原始信号）
s1 = (df['close'] < ma200).fillna(False)
s2 = (adx < 20).fillna(False)
s3 = (atr_pct > 0.75).fillna(False)
s4 = (vol_zscore > 2).fillna(False)
# 新增候选信号
s5 = (ret_21 < -0.12).fillna(False)  # 3周跌幅
s6 = (ret_30 < 0).fillna(False)       # 30日动量

combos = [
    ("原始：MA200+ADX<20+ATR75%+VolZ>2，>=2/4",   [s1,s2,s3,s4], 2),
    ("原始：MA200+ADX<20+ATR75%+VolZ>2，>=3/4",   [s1,s2,s3,s4], 3),
    ("MA200+ADX<25+ATR75%+VolZ>2，>=2/4",          [s1,(adx<25).fillna(False),s3,s4], 2),
    ("MA200+ADX<20+ATR65%+VolZ>1.5，>=2/4",        [s1,s2,(atr_pct>0.65).fillna(False),(vol_zscore>1.5).fillna(False)], 2),
    ("MA200+ADX<20+ATR75%+跌幅12%，>=2/4",         [s1,s2,s3,s5], 2),
    ("MA200+ADX<20+ATR75%+跌幅12%，>=2/4（宽松）", [s1,s2,s3,s5], 2),
    ("5信号：+30日动量，>=3/5",                     [s1,s2,s3,s4,s6], 3),
    ("5信号：+跌幅12%，>=3/5",                      [s1,s2,s3,s5,s6], 3),
    ("纯价格：MA200+跌幅12%+30日动量，>=2/3",       [s1,s5,s6], 2),
]

best_score = -999
best_combo = None

for name, sigs, threshold in combos:
    score = sum(s.astype(int) for s in sigs)
    regime = (score >= threshold).astype(int)
    reg = regime[valid_idx]
    off_mask = reg == 1
    on_mask = reg == 0
    if off_mask.sum() < 20:
        continue
    pct = off_mask.mean() * 100
    ret_off = r[off_mask].mean() * 100
    ret_on = r[on_mask].mean() * 100
    flips = (reg.diff().abs() > 0).sum()
    diff = ret_off - ret_on
    marker = " <-- 最优" if diff < best_score else ""
    if diff < best_score:
        best_score = diff
        best_combo = name
    print(f"  {name:<43} {pct:>8.1f}% {ret_off:>+11.3f}% {ret_on:>+11.3f}% {flips:>8} {diff:>+7.3f}%{marker}")

# ─── 3. 平滑处理效果 ──────────────────────────────────────────────────────────
print("\n=== 平滑处理（减少翻转）===")
print(f"{'方案':<40} {'risk_off%':>9} {'off日均收益':>12} {'on日均收益':>12} {'翻转次数':>8}")
print("-" * 85)

# 原始方案A
base_score = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)
base_regime = (base_score >= 2).astype(int)

for smooth_days in [0, 3, 5, 7]:
    if smooth_days == 0:
        reg = base_regime[valid_idx]
        label = "无平滑（原始）"
    else:
        # 滚动N天内有>=1天触发才维持状态（粘性）
        smoothed = base_regime.rolling(smooth_days).max().fillna(0).astype(int)
        reg = smoothed[valid_idx]
        label = f"粘性平滑（{smooth_days}天窗口）"
    off_mask = reg == 1
    on_mask = reg == 0
    pct = off_mask.mean() * 100
    ret_off = r[off_mask].mean() * 100 if off_mask.sum() > 0 else 0
    ret_on = r[on_mask].mean() * 100 if on_mask.sum() > 0 else 0
    flips = (reg.diff().abs() > 0).sum()
    print(f"  {label:<38} {pct:>8.1f}% {ret_off:>+11.3f}% {ret_on:>+11.3f}% {flips:>8}")

