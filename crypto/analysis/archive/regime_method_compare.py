"""
三种市场制度识别方案对比（2026-05-27）

方案A：规则投票（BTC MA200 + ADX + ATR百分位 + 资金费率）
方案B：HMM（3状态，Binance长历史数据）
方案C：GARCH波动率制度（ATR百分位连续概率，MSGARCH简化版）

结论：
- 方案A（规则投票）：risk_off 日均收益 -0.021%，risk_on +0.193%，方向正确 ✓
- 方案B（HMM）：risk_off 日均收益 +0.067%，方向错误 ✗（OKX数据只有~300天全是牛市）
- 方案C（GARCH）：risk_off > risk_on，方向错误 ✗
- 最终结论：双向策略中大跌=做空盈利，BTC价格信号无效；真正的敌人是横盘（低ADX+低ATR）

用法：
  python regime_method_compare.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ccxt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# ─── 1. 拉数据（Binance，2020至今）─────────────────────────────────────────────

print("拉取 BTC 日线数据（Binance，2020-01-01 至今）...")
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
print(f"获取到 {len(df)} 根日线，{df.index[0].date()} ~ {df.index[-1].date()}")

returns = df['close'].pct_change().fillna(0)

# ─── 2. 方案A：规则投票 ────────────────────────────────────────────────────────

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
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

print("\n计算方案A（规则投票）...")
ma200 = df['close'].rolling(200).mean()
adx = compute_adx(df, 14)

atr_raw = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1).rolling(14).mean()
atr_pct = atr_raw.rolling(252).rank(pct=True)  # 历史百分位

# 资金费率用波动率Z-score替代（无历史资金费率数据）
vol_20 = returns.rolling(20).std()
vol_zscore = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()

score_a = pd.Series(0.0, index=df.index)
score_a += (df['close'] < ma200).astype(float)       # 信号1：价格低于MA200
score_a += (adx < 20).astype(float)                   # 信号2：ADX无趋势
score_a += (atr_pct > 0.75).astype(float)             # 信号3：ATR高波动
score_a += (vol_zscore > 2).astype(float)             # 信号4：波动率异常

prob_a = (score_a / 4).fillna(0)
regime_a = (prob_a >= 0.5).astype(int)  # >=2个信号触发 = risk_off

# ─── 3. 方案B：HMM（3状态）────────────────────────────────────────────────────

print("计算方案B（HMM 3状态）...")
vol_20_raw = returns.rolling(20).std().fillna(returns.std())
features = np.column_stack([returns.values, vol_20_raw.values])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 多次运行取最稳定结果
best_score = -np.inf
best_model = None
for seed in range(20):
    try:
        m = hmm.GaussianHMM(n_components=3, covariance_type='full',
                             n_iter=300, random_state=seed)
        m.fit(features_scaled)
        if m.monitor_.converged and m.score(features_scaled) > best_score:
            best_score = m.score(features_scaled)
            best_model = m
    except:
        pass

states_b = best_model.predict(features_scaled)
# 按均值收益排序：最低均值收益的状态 = risk_off
mean_returns_by_state = [returns.values[states_b == s].mean() for s in range(3)]
risk_off_state = int(np.argmin(mean_returns_by_state))
chop_state = int(np.argsort(mean_returns_by_state)[1])  # 中间状态

# 输出后验概率（risk_off 概率）
posteriors = best_model.predict_proba(features_scaled)
prob_b = pd.Series(posteriors[:, risk_off_state], index=df.index)
# risk_off = 最差状态概率 > 0.5，或中间+最差 > 0.7
regime_b = ((posteriors[:, risk_off_state] + posteriors[:, chop_state]) > 0.7).astype(int)
regime_b = pd.Series(regime_b, index=df.index)

# ─── 4. 方案C：GARCH波动率制度（连续概率）────────────────────────────────────

print("计算方案C（GARCH波动率制度）...")
try:
    from arch import arch_model
    # 拟合 GJR-GARCH(1,1)
    ret_pct = returns * 100  # arch 库需要百分比收益率
    garch_model = arch_model(ret_pct, vol='Garch', p=1, o=1, q=1, dist='t')
    garch_fit = garch_model.fit(disp='off', show_warning=False)
    cond_vol = garch_fit.conditional_volatility / 100  # 转回小数

    # 用历史百分位转换为概率
    vol_pct_rank = cond_vol.rolling(252).rank(pct=True).fillna(
        cond_vol.expanding().rank(pct=True)
    )
    prob_c = vol_pct_rank  # 直接用百分位作为 risk_off 概率
    regime_c = (prob_c > 0.75).astype(int)
except Exception as e:
    print(f"  GARCH 失败：{e}，用 ATR 百分位替代")
    prob_c = atr_pct.fillna(0)
    regime_c = (prob_c > 0.75).astype(int)

# ─── 5. 评估 ──────────────────────────────────────────────────────────────────

# 只用有足够历史的数据（MA200需要200天）
valid_idx = df.index[200:]
r = returns[valid_idx]

def evaluate(regime, prob, name):
    reg = regime[valid_idx]
    p = prob[valid_idx]

    risk_off_mask = reg == 1
    risk_on_mask = reg == 0

    # 基础统计
    n_risk_off = risk_off_mask.sum()
    n_risk_on = risk_on_mask.sum()
    pct_risk_off = n_risk_off / len(reg) * 100

    # risk_off 期间实际表现
    r_off = r[risk_off_mask]
    r_on = r[risk_on_mask]

    mean_ret_off = r_off.mean() * 100 if len(r_off) > 0 else 0
    mean_ret_on = r_on.mean() * 100 if len(r_on) > 0 else 0
    down_rate_off = (r_off < 0).mean() * 100 if len(r_off) > 0 else 0  # 下跌天数占比

    # 信号翻转次数（稳定性）
    flips = (reg.diff().abs() > 0).sum()

    # 概率分布
    prob_mean_off = p[risk_off_mask].mean() if risk_off_mask.sum() > 0 else 0
    prob_mean_on = p[risk_on_mask].mean() if risk_on_mask.sum() > 0 else 0

    # 识别出的主要 risk_off 时期
    # 找连续 risk_off 段
    segments = []
    in_seg = False
    start = None
    for date, val in reg.items():
        if val == 1 and not in_seg:
            in_seg = True
            start = date
        elif val == 0 and in_seg:
            in_seg = False
            segments.append((start, date, (date - start).days))
    if in_seg:
        segments.append((start, reg.index[-1], (reg.index[-1] - start).days))

    # 取最长的5个
    top_segs = sorted(segments, key=lambda x: x[2], reverse=True)[:5]

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  risk_off 天数：{n_risk_off} / {len(reg)}（{pct_risk_off:.1f}%）")
    print(f"  risk_off 期间日均收益：{mean_ret_off:+.3f}%")
    print(f"  risk_off 期间下跌天数占比：{down_rate_off:.1f}%")
    print(f"  risk_on  期间日均收益：{mean_ret_on:+.3f}%")
    print(f"  信号翻转次数：{flips}（越少越稳定）")
    print(f"  概率均值（risk_off时）：{prob_mean_off:.2f}")
    print(f"  概率均值（risk_on时）：{prob_mean_on:.2f}")
    print(f"\n  主要 risk_off 时期（最长5段）：")
    for s, e, d in top_segs:
        btc_ret = (df['close'][e] / df['close'][s] - 1) * 100 if s in df.index and e in df.index else 0
        print(f"    {s.date()} ~ {e.date()}（{d}天）BTC实际涨跌：{btc_ret:+.1f}%")

    return {
        'name': name,
        'pct_risk_off': pct_risk_off,
        'mean_ret_off': mean_ret_off,
        'down_rate_off': down_rate_off,
        'mean_ret_on': mean_ret_on,
        'flips': flips,
        'prob_separation': prob_mean_off - prob_mean_on,
    }

results = []
results.append(evaluate(regime_a, prob_a, "方案A：规则投票"))
results.append(evaluate(regime_b, prob_b, "方案B：HMM 3状态"))
results.append(evaluate(regime_c, prob_c, "方案C：GARCH波动率制度"))

# ─── 6. 汇总对比 ──────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  汇总对比")
print(f"{'='*55}")
print(f"{'指标':<25} {'方案A':>10} {'方案B':>10} {'方案C':>10}")
print(f"{'-'*55}")
metrics = [
    ('risk_off占比(%)', 'pct_risk_off', '{:.1f}'),
    ('risk_off日均收益(%)', 'mean_ret_off', '{:+.3f}'),
    ('risk_off下跌天数(%)', 'down_rate_off', '{:.1f}'),
    ('risk_on日均收益(%)', 'mean_ret_on', '{:+.3f}'),
    ('信号翻转次数', 'flips', '{:.0f}'),
    ('概率区分度', 'prob_separation', '{:.2f}'),
]
for label, key, fmt in metrics:
    vals = [fmt.format(r[key]) for r in results]
    print(f"  {label:<23} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

print(f"\n  说明：")
print(f"  - risk_off日均收益越负 = 识别越准确（确实是坏时期）")
print(f"  - risk_off下跌天数越高 = 识别越准确")
print(f"  - 信号翻转次数越少 = 信号越稳定")
print(f"  - 概率区分度越高 = 两种制度区分越清晰")

