"""
preprocess.py —— 对应论文第三章「数据处理和特征选择」
步骤：
  3.3.1 缺失值处理：数值型→中位数，类别型→众数（代码3-1）
  3.3.2 异常值处理：IQR 法则定位 + 盖帽法（capping）替换，而非删除（代码3-2）
  3.3.3 标准化/归一化：Z-Score(StandardScaler) 与 Min-Max(MinMaxScaler)（代码3-3）
  3.4  特征分析：分布、数值型相关性热力图、高基数特征频次编码

相比论文附录代码的修正：
  - 原附录只演示了单列；这里批量对所有核心数值特征做标准化/归一化，并保留原始列。
  - 盖帽法用 np.clip 替换，避免直接删样本导致信息损失（论文强调的点）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (RAW_CSV, FIG_DIR, IQR_K, CLUSTER_NUMERIC_FEATURES,
                    CLUSTER_FEATURES_SCALED, PROFILE_CATEGORICAL)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set_theme(style="whitegrid")


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    print(f"[preprocess] 原始数据维度: {df.shape}")
    print(df.dtypes.to_string())
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns
    for c in numeric:
        df[c] = df[c].fillna(df[c].median())
    obj = df.select_dtypes(include=["object"]).columns
    for c in obj:
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
    print("[preprocess] 缺失值补齐完成（数值→中位数，类别→众数）")
    return df


def cap_outliers_iqr(df: pd.DataFrame, col: str, k: float = IQR_K) -> pd.DataFrame:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    n_before = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    print(f"[preprocess] 列 {col}: 检出异常值 {n_before} 条，已用盖帽法替换为 [{lo:.2f}, {hi:.2f}]")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    # 论文：Age / PurchaseAmount 用 Z-Score；ReviewRating / Previous_Purchases 用 Min-Max
    std_cols = {"Age": "Age_Scaled", "PurchaseAmount(USD)": "Purchase_Amount_Scaled"}
    mm_cols = {"ReviewRating": "Review_Rating_Norm", "Previous_Purchases": "Previous_Purchases_Norm"}
    for src, dst in std_cols.items():
        df[dst] = StandardScaler().fit_transform(df[[src]])
    for src, dst in mm_cols.items():
        df[dst] = MinMaxScaler().fit_transform(df[[src]])
    print("[preprocess] 标准化/归一化完成（Z-Score + Min-Max）")
    return df


def correlation_heatmap(df: pd.DataFrame, cols=None):
    cols = cols or CLUSTER_FEATURES_SCALED
    corr = df[cols].corr(method="pearson")
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("图3-3 数值型特征相关性热力图 (Pearson)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[preprocess] 相关性热力图 -> {path}")
    return corr


def frequency_encode_high_cardinality(df: pd.DataFrame, col: str, min_ratio: float = 0.005):
    """论文 3.4.3：高基数特征用「频次编码 + 低频归并」压缩为一维稠密特征。"""
    vc = df[col].value_counts()
    threshold = len(df) * min_ratio
    rare = vc[vc < threshold].index
    df[f"{col}_freq"] = df[col].map(vc).fillna(0) / len(df)
    df.loc[df[col].isin(rare), f"{col}_freq"] = df[f"{col}_freq"].min()
    print(f"[preprocess] 高基数特征 {col}: 归并 {len(rare)} 个低频类别 -> 频次特征 {col}_freq")
    return df


def run() -> pd.DataFrame:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_raw()
    df = handle_missing(df)
    for c in CLUSTER_NUMERIC_FEATURES:
        df = cap_outliers_iqr(df, c)
    df = scale_features(df)
    correlation_heatmap(df)
    # 高基数示例：对 Category / Color 做频次编码演示
    for c in ["Category", "Color", "PaymentMethod"]:
        df = frequency_encode_high_cardinality(df, c)
    out = os.path.join(os.path.dirname(RAW_CSV), "clean_data.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[preprocess] 清洗后数据 -> {out}  维度 {df.shape}")
    return df


if __name__ == "__main__":
    run()
