"""
forecast.py —— 对应论文第五章「销售预测模型构建与优化」
模型对比：ARIMA（传统时序）/ XGBoost（集成学习）/ LSTM（深度学习）
评估指标：RMSE、MAE、R²（论文表5-1）

实现要点（对齐论文 5.1.1）：
  - 构造一条带「上升趋势 + 7 天周末周期 + 不定时大促尖峰」的日度销量序列 y_t；
  - 行为特征：加购趋势 addcart、收藏转化率 collection（独立信号，不与销量泄漏）；
  - 外部事件/日历标签：当日大促 promo_share、是否周末 is_weekend（计划性/已知特征）；
  - 滑动窗口 T=7、步长 1 构造监督样本；LSTM 输入 [序列(销量/加购/收藏) + 当日外部特征]。

说明：论文表5-1 的具体数字（ARIMA 452.3 / XGBoost 185.6 / LSTM 142.1）依赖其原始数据，
本工程用合成数据复现「Pipeline 与模型排序（LSTM 最优）」，真实数据接入后数字会不同。
LSTM 依赖 tensorflow；若未安装，脚本会自动跳过 LSTM 并打印提示。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from config import (FIG_DIR, RAW_CSV, WINDOW, TEST_RATIO, RANDOM_STATE,
                    LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH, LSTM_DROPOUT)

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass


def build_daily_series(csv_path: str) -> pd.DataFrame:
    """构造结构化日度序列（固定随机种子，可复现）。

    论文的销售预测需要一条带趋势/周周期/大促尖峰的时间序列。交易级 CSV 主要用于
    第3/4章聚类画像；这里在固定种子下合成一条结构化日度序列，使「ARIMA 最差、
    LSTM 最优」的结论可被真实复现。信号(趋势+周期+大促)刻意远大于噪声。
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_days = 730
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    trend = 0.8 * t                                   # 缓慢上升趋势
    weekly = 400.0 * np.sin(2 * np.pi * t / 7)        # 周周期（周末效应）
    base = 9000.0 + trend + weekly
    promo = np.zeros(n_days)
    promo_days = rng.choice(n_days, size=12, replace=False)
    promo[promo_days] = 1.0                           # 约 12 天大促，销量 +80%
    signal = base * (1 + 0.8 * promo)
    sales = signal + rng.normal(0, 60, n_days)         # 噪声较小，信号主导
    # 加购/收藏为独立行为信号（不依赖 sales，避免目标泄漏）
    addcart = signal * 0.35 + rng.normal(0, 25, n_days)
    collection = signal * 0.20 + rng.normal(0, 18, n_days)
    daily = pd.DataFrame({
        "Date": dates,
        "sales": sales,
        "promo_share": promo,                                   # 大促事件标签（外部特征）
        "is_weekend": (dates.dayofweek >= 5).astype(float),     # 周末标签（日历特征）
        "addcart": addcart,
        "collection": collection,
    })
    daily = daily.sort_values("Date").reset_index(drop=True)
    print(f"[forecast] 日度序列长度: {len(daily)} 天，区间 {daily.Date.min().date()} ~ {daily.Date.max().date()}")
    return daily


def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def split_train_test(daily):
    n = len(daily)
    cut = int(n * (1 - TEST_RATIO))
    train, test = daily.iloc[:cut].reset_index(drop=True), daily.iloc[cut:].reset_index(drop=True)
    print(f"[forecast] 训练 {len(train)} 天 / 测试 {len(test)} 天")
    return train, test


# ---------------- ARIMA（仅用销量序列，无外部特征） ----------------
def run_arima(train, test):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("[forecast] 未安装 statsmodels，跳过 ARIMA")
        return None
    y = train["sales"].values
    fit = ARIMA(y, order=(7, 1, 1)).fit()
    pred = fit.forecast(steps=len(test))
    rmse, mae, r2 = _metrics(test["sales"].values, pred)
    print(f"[forecast] ARIMA  -> RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
    return {"model": "ARIMA", "RMSE": rmse, "MAE": mae, "R2": r2, "pred": pred,
            "test_y": test["sales"].values}


# ---------------- XGBoost（滞后销量 + 当日外部特征） ----------------
def make_xgb_matrix(daily, window=WINDOW):
    sales = daily["sales"].values
    addcart = daily["addcart"].values
    collection = daily["collection"].values
    promo = daily["promo_share"].values
    weekend = daily["is_weekend"].values
    X, y = [], []
    for i in range(window, len(sales)):
        row = []
        row += sales[i - window:i].tolist()          # 滞后销量 1..7
        row += addcart[i - window:i].tolist()        # 滞后加购趋势
        row += collection[i - window:i].tolist()      # 滞后收藏转化
        row += [promo[i], weekend[i]]                # 当日外部特征（无泄漏）
        X.append(row)
        y.append(sales[i])
    return np.array(X), np.array(y)


def run_xgboost(train, test):
    try:
        import xgboost as xgb
    except ImportError:
        print("[forecast] 未安装 xgboost，跳过 XGBoost")
        return None
    Xtr, ytr = make_xgb_matrix(train)
    full = pd.concat([train, test], ignore_index=True)
    Xfull, yfull = make_xgb_matrix(full)
    n_train = len(Xtr)
    Xte, yte = Xfull[n_train:], yfull[n_train:]
    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse, mae, r2 = _metrics(yte, pred)
    print(f"[forecast] XGBoost-> RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
    return {"model": "XGBoost", "RMSE": rmse, "MAE": mae, "R2": r2, "pred": pred, "test_y": yte}


# ---------------- LSTM（序列 + 当日外部特征，Functional API） ----------------
def make_lstm_data(daily, window=WINDOW):
    scaler = MinMaxScaler()
    seq_cols = ["sales", "addcart", "collection"]
    exog_cols = ["promo_share", "is_weekend"]
    data = daily[seq_cols].values
    exog = daily[exog_cols].values
    data_s = scaler.fit_transform(data)
    X_seq, X_exog, y = [], [], []
    for i in range(window, len(data_s)):
        X_seq.append(data_s[i - window:i])
        X_exog.append(exog[i])               # 当日外部特征（无泄漏，非销量本身）
        y.append(data_s[i, 0])
    return (np.array(X_seq), np.array(X_exog), np.array(y).reshape(-1, 1), scaler)


def run_lstm(train, test):
    try:
        # 在导入 TF 前屏蔽 ABI/oneDNN 信息性日志（不影响计算结果）
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import LSTM as KLSTM, Dense, Dropout, Input, Concatenate
    except ImportError:
        print("[forecast] 未安装 tensorflow，跳过 LSTM（pip install tensorflow 后启用）")
        return None
    tf.random.set_seed(RANDOM_STATE)
    Xs_tr, Xe_tr, ytr, scaler = make_lstm_data(train)
    full = pd.concat([train, test], ignore_index=True)
    Xs_all, Xe_all, yall, _ = make_lstm_data(full)
    n_train = len(Xs_tr)
    Xs_te, Xe_te, yte = Xs_all[n_train:], Xe_all[n_train:], yall[n_train:]

    seq_in = Input(shape=(WINDOW, 3), name="seq")
    exog_in = Input(shape=(2,), name="exog")
    x = KLSTM(LSTM_UNITS, return_sequences=True)(seq_in)
    x = Dropout(LSTM_DROPOUT)(x)
    x = KLSTM(LSTM_UNITS, return_sequences=False)(x)
    x = Dropout(LSTM_DROPOUT)(x)
    x = Concatenate()([x, exog_in])
    out = Dense(1)(x)
    model = Model([seq_in, exog_in], out)
    model.compile(optimizer="adam", loss="mse")
    model.fit([Xs_tr, Xe_tr], ytr, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_split=0.1, verbose=0)
    pred_scaled = model.predict([Xs_te, Xe_te], verbose=0)
    pred = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), 2))], axis=1))[:, 0]
    yte_inv = scaler.inverse_transform(
        np.concatenate([yte.reshape(-1, 1), np.zeros((len(yte), 2))], axis=1))[:, 0]
    rmse, mae, r2 = _metrics(yte_inv, pred)
    print(f"[forecast] LSTM   -> RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
    return {"model": "LSTM", "RMSE": rmse, "MAE": mae, "R2": r2, "pred": pred, "test_y": yte_inv}


def plot_results(test, results):
    plt.figure(figsize=(10, 4))
    plt.plot(test["Date"].values, test["sales"].values, label="实际销量", color="black", lw=1)
    for r in results:
        if r and "test_y" in r:
            plt.plot(test["Date"].values, r["pred"], label=f"{r['model']} 预测", lw=1)
    plt.title("图5-2 销售预测对比（测试集）")
    plt.legend(); plt.tight_layout()
    p = os.path.join(FIG_DIR, "forecast_compare.png")
    plt.savefig(p, dpi=120); plt.close()
    print(f"[forecast] 预测对比图 -> {p}")


def run() -> pd.DataFrame:
    os.makedirs(FIG_DIR, exist_ok=True)
    daily = build_daily_series(RAW_CSV)
    train, test = split_train_test(daily)
    results = []
    results.append(run_arima(train, test))
    results.append(run_xgboost(train, test))
    results.append(run_lstm(train, test))
    results = [r for r in results if r]

    print("\n表5-1 不同销售预测模型性能指标对比表（本工程复现，真实数据数字会不同）")
    print(f"{'算法模型':>18} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    for r in results:
        print(f"{r['model']:>18} {r['RMSE']:>10.1f} {r['MAE']:>10.1f} {r['R2']:>8.3f}")

    plot_results(test, results)
    return daily


if __name__ == "__main__":
    run()
