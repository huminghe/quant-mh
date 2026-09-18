"""
Qlib GRU（Alpha360）深度序列模型 walk-forward 验证 —— 判断是否值得投入

背景：单一切分（train2016-21/valid22-23/test24-26，seed=42）跑出 test ICIR 0.43、
train ICIR 1.06，train>>valid/test 存在明显衰减，怀疑是过拟合或撞上了单一窗口的
运气。用户指出验证方法不严谨（单一切分、单一种子、无正则化、无基线对比、切分点
可能有 label 泄漏），要求重新设计验证方案。

本脚本实现：
1. 两个不重叠的滚动窗口（WINDOWS 字典），每个窗口的 train/valid/test 切分点之间
   留 LABEL_HORIZON 个交易日的 purge gap（label 是未来20日收益，切分点前20天的
   样本会用到切分点后的价格，不隔开会有信息泄漏）
2. 每个窗口跑 2 个随机种子（SEEDS），共 4 次独立训练，用于分离"初始化噪声"和
   "窗口本身效应"
3. Adam 优化器手动加 weight_decay（qlib 的 GRU 类 __init__ 不支持传参，训练前
   直接覆盖 model.train_optimizer）——原脚本唯一的正则化只有 dropout=0.1 和
   early_stop=5，早停在 Epoch1 就锁定，说明防过拟合力度不够
4. 每个窗口同时算一个 20 日动量因子的 IC 作为基线（复用 fetch_index_members.py
   的 point-in-time 成分股工具），几秒钟出结果，用于判断 GRU 是否只是绕远路
   重新发现了已知的线性因子

判定标准：多数窗口里 GRU 的 valid+test 段 IC 同时满足"稳定为正"且"明显超过
同窗口动量基线"，才算深度模型方向有效；否则止损。

用法（需要用 venv_qlib 独立环境执行）：
  cd a_stock/backtest
  ../../venv_qlib/bin/python3 qlib_gru_poc_walkforward.py
"""

import pathlib

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha360
from qlib.contrib.model.pytorch_gru import GRU
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R
from scipy.stats import spearmanr
import torch.optim as optim

BASE_DIR = pathlib.Path(__file__).parent.parent
PROVIDER_URI = str(BASE_DIR / "qlib_data" / "stock_day")
CACHE_DIR = BASE_DIR / "qlib_data"
MEMBERS_DATA_DIR = BASE_DIR / "data"
STOCK_DAILY_DIR = MEMBERS_DATA_DIR / "stock_daily"


def load_close_panel(min_coverage: float = 0.5) -> pd.DataFrame:
    """
    读取中证500全部出现过成分股的收盘价面板（宽格式），index=trade_date。
    从 fetch_index_members.py 内联而来，避免 import 该模块触发其顶层
    `import tushare`（venv_qlib 环境未装 tushare，只是借用这两个纯 pandas 函数）。
    """
    members = pd.read_parquet(MEMBERS_DATA_DIR / "hs500_members.parquet")
    codes = sorted(members["con_code"].unique().tolist())

    frames = {}
    for code in codes:
        path = STOCK_DAILY_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["trade_date", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            frames[code] = df.set_index("trade_date")["close"]

    panel = pd.DataFrame(frames).sort_index()
    threshold = int(len(panel.columns) * min_coverage)
    last_complete = panel.index[panel.notna().sum(axis=1) >= threshold][-1]
    return panel.loc[:last_complete]


def load_members_pit(date: pd.Timestamp, members_file: pathlib.Path) -> list:
    """指定日期的 point-in-time 中证500成分股列表（取 <=date 的最近月末快照）"""
    members = pd.read_parquet(members_file)
    members["trade_date"] = pd.to_datetime(members["trade_date"])
    valid = members[members["trade_date"] <= date]
    if valid.empty:
        return []
    latest = valid["trade_date"].max()
    return members[members["trade_date"] == latest]["con_code"].tolist()


def log(msg: str) -> None:
    """实时打印+flush，避免管道缓冲导致看不到训练进度"""
    print(msg, flush=True)


LABEL_HORIZON = 20  # 未来20日收益，与反转因子实验的 label 尺度对齐
PURGE = LABEL_HORIZON  # 切分点两侧各留20个交易日 gap，避免 label 跨切分点泄漏

# 两个不重叠的滚动窗口。每个窗口内 train_end -> valid_start、valid_end -> test_start
# 都已经手工空出约1个月（>=20个交易日），近似 purge gap。
WINDOWS = {
    "A": dict(
        start_time="2015-06-01",
        fit_start="2016-01-01", fit_end="2019-11-30",
        valid_start="2020-01-01", valid_end="2020-11-30",
        test_start="2021-01-01", test_end="2022-06-30",
        end_time="2022-06-30",
    ),
    "B": dict(
        start_time="2017-06-01",
        fit_start="2018-01-01", fit_end="2021-11-30",
        valid_start="2022-01-01", valid_end="2022-11-30",
        test_start="2023-01-01", test_end="2024-06-30",
        end_time="2024-06-30",
    ),
}

SEEDS = [42, 123]
WEIGHT_DECAY = 1e-4  # qlib GRU 默认不加 L2 正则，手动补上


def compute_ic_series(pred: pd.Series, label: pd.Series) -> pd.Series:
    """按日期分组计算截面 Rank IC"""
    df = pd.concat([pred, label], axis=1)
    df.columns = ["pred", "label"]
    df = df.dropna()

    def _ic(g):
        if len(g) < 20:
            return np.nan
        ic, _ = spearmanr(g["pred"], g["label"])
        return ic

    return df.groupby(level="datetime").apply(_ic)


def get_or_build_handler(window_key: str, cfg: dict) -> DataHandlerLP:
    cache_path = CACHE_DIR / f"gru_poc_handler_{window_key}.pkl"
    if cache_path.exists():
        log(f"[窗口{window_key}] 发现缓存的 handler，跳过特征计算")
        return DataHandlerLP.load(cache_path)

    log(f"[窗口{window_key}] 未发现缓存，开始计算 Alpha360 特征（预计约15分钟）...")
    label_config = (
        [f"Ref($close, -{LABEL_HORIZON})/$close - 1"],
        ["LABEL0"],
    )
    handler = Alpha360(
        instruments="csi500",
        start_time=cfg["start_time"],
        end_time=cfg["end_time"],
        fit_start_time=cfg["fit_start"],
        fit_end_time=cfg["fit_end"],
        label=label_config,
    )
    handler.to_pickle(cache_path, dump_all=True)
    log(f"[窗口{window_key}] 特征计算完成，已缓存到 {cache_path}")
    return handler


def run_gru(window_key: str, cfg: dict, seed: int, handler: DataHandlerLP) -> dict:
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (cfg["fit_start"], cfg["fit_end"]),
            "valid": (cfg["valid_start"], cfg["valid_end"]),
            "test": (cfg["test_start"], cfg["test_end"]),
        },
    )

    log(f"[窗口{window_key} seed={seed}] 开始训练 GRU ...")
    model = GRU(
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        n_epochs=15,
        lr=0.001,
        early_stop=5,
        batch_size=2000,
        metric="loss",
        loss="mse",
        GPU=-1,
        seed=seed,
    )
    # qlib GRU.__init__ 不支持 weight_decay 参数，训练前手动换成带 L2 正则的 Adam
    model.train_optimizer = optim.Adam(
        model.gru_model.parameters(), lr=model.lr, weight_decay=WEIGHT_DECAY
    )

    with R.start(experiment_name=f"gru_poc_{window_key}_{seed}"):
        model.fit(dataset)
    log(f"[窗口{window_key} seed={seed}] 训练完成，开始评估 valid/test 段...")

    result = {"window": window_key, "seed": seed}
    for seg in ["valid", "test"]:
        pred = model.predict(dataset, segment=seg)
        label = dataset.prepare(seg, col_set="label", data_key=DataHandlerLP.DK_R)
        label = label.iloc[:, 0]
        label.name = "label"

        ic_series = compute_ic_series(pred, label).dropna()
        icir = ic_series.mean() / ic_series.std() if ic_series.std() > 1e-8 else np.nan

        result[f"{seg}_n"] = len(ic_series)
        result[f"{seg}_ic_mean"] = ic_series.mean()
        result[f"{seg}_icir"] = icir
        result[f"{seg}_pos_pct"] = (ic_series > 0).mean()

        log(
            f"[窗口{window_key} seed={seed}] {seg}段 截面数={len(ic_series)} "
            f"IC均值={ic_series.mean():.4f} ICIR={icir:.4f} "
            f"IC>0占比={(ic_series > 0).mean():.1%}"
        )

    return result


def momentum_baseline(window_key: str, cfg: dict) -> dict:
    """20日动量因子（简单价格序列线性信号）在同一窗口 valid/test 段的 IC，作为基线"""
    close_panel = load_close_panel()
    result = {"window": window_key}

    for seg, (seg_start, seg_end) in [
        ("valid", (cfg["valid_start"], cfg["valid_end"])),
        ("test", (cfg["test_start"], cfg["test_end"])),
    ]:
        idx = close_panel.index
        dates = idx[(idx >= seg_start) & (idx <= seg_end)]
        records = []
        for d in dates:
            pos = idx.searchsorted(d)
            if pos < LABEL_HORIZON or pos + LABEL_HORIZON >= len(idx):
                continue
            pit_members = load_members_pit(d, members_file=MEMBERS_DATA_DIR / "hs500_members.parquet")
            if not pit_members:
                continue
            available = [c for c in pit_members if c in close_panel.columns]
            if len(available) < 80:
                continue

            mom = close_panel[available].iloc[pos] / close_panel[available].iloc[pos - LABEL_HORIZON] - 1
            mom = mom.dropna()
            fwd_ret = (
                close_panel[available].iloc[pos + LABEL_HORIZON] / close_panel[available].iloc[pos] - 1
            ).dropna()
            common = mom.index.intersection(fwd_ret.index)
            if len(common) < 50:
                continue
            ic, _ = spearmanr(mom[common], fwd_ret[common])
            records.append(ic)

        ic_arr = pd.Series(records).dropna()
        icir = ic_arr.mean() / ic_arr.std() if ic_arr.std() > 1e-8 else np.nan
        result[f"{seg}_n"] = len(ic_arr)
        result[f"{seg}_ic_mean"] = ic_arr.mean()
        result[f"{seg}_icir"] = icir
        log(
            f"[窗口{window_key} 动量基线] {seg}段 截面数={len(ic_arr)} "
            f"IC均值={ic_arr.mean():.4f} ICIR={icir:.4f}"
        )

    return result


def main():
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    gru_results = []
    momentum_results = []

    for window_key, cfg in WINDOWS.items():
        handler = get_or_build_handler(window_key, cfg)
        for seed in SEEDS:
            gru_results.append(run_gru(window_key, cfg, seed, handler))
        momentum_results.append(momentum_baseline(window_key, cfg))

    log("\n" + "=" * 70)
    log("=== 汇总：GRU walk-forward 结果 vs 20日动量因子基线 ===")
    log("=" * 70)
    for r in gru_results:
        log(
            f"GRU 窗口{r['window']} seed={r['seed']}: "
            f"valid ICIR={r['valid_icir']:.4f} (n={r['valid_n']}, IC均值={r['valid_ic_mean']:.4f}) | "
            f"test ICIR={r['test_icir']:.4f} (n={r['test_n']}, IC均值={r['test_ic_mean']:.4f})"
        )
    for r in momentum_results:
        log(
            f"动量基线 窗口{r['window']}: "
            f"valid ICIR={r['valid_icir']:.4f} (IC均值={r['valid_ic_mean']:.4f}) | "
            f"test ICIR={r['test_icir']:.4f} (IC均值={r['test_ic_mean']:.4f})"
        )

    log("\n判定标准：多数窗口里 GRU 的 valid+test 段 IC 同时满足"
        "\"稳定为正\"且\"明显超过同窗口动量基线\"，才算深度模型方向有效；"
        "\n否则（方向不稳定/不superior于线性动量因子/种子间大幅波动）判定为"
        "过拟合或对已知线性信号的复杂重复，止损不再投入。")


if __name__ == "__main__":
    main()
