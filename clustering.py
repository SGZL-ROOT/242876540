"""
clustering.py —— 对应论文第四章「用户画像的构建与分析」
步骤：
  4.2 用户分群：手肘法(SSE) + 轮廓系数 选 K；最终 K=3（论文表4-1）
  4.3 画像可视化：雷达图、各群体年龄分布箱线图、性别占比、商品类别/颜色/支付方式/订阅率

相比论文附录代码的修正：
  - 原附录用 kmeans.fit_transform(features) 当作标签，这是错误的（fit_transform 返回的是
    到各簇中心的距离矩阵）。正确做法是 fit_predict 得到簇标签。本脚本已修正。
  - 额外计算并打印表4-1 的 SSE 与平均轮廓系数。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (FIG_DIR, KMIN, KMAX, BEST_K, RANDOM_STATE,
                    CLUSTER_FEATURES_SCALED, PROFILE_CATEGORICAL)

sns.set_theme(style="whitegrid")

# 尝试设置支持中文的字体，避免标题出现方块（□）。若环境中无中文字体则跳过（仅影响显示）。
for _f in ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "WenQuanYi Micro Hei"]:
    try:
        plt.rcParams["font.family"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# 轮廓系数为 O(n^2)，10 万样本上极慢；按业界惯例用子样本估算（论文也未说明全量计算）。
SILHOUETTE_SAMPLE = 10_000


def elbow_and_silhouette(df: pd.DataFrame, features):
    X = df[features].values
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    sub = min(SILHOUETTE_SAMPLE, n)
    idx = rng.choice(n, sub, replace=False)
    Xs = X[idx]
    sse, scores = [], []
    ks = list(range(KMIN, KMAX + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sse.append(km.inertia_)
        # 轮廓系数仅在子样本上计算，避免 O(n^2) 在 10 万样本上卡死
        labels_sub = km.predict(Xs) if sub < n else labels
        scores.append(silhouette_score(Xs, labels_sub))
    # 打印表4-1
    print("\n表4-1 不同聚类数 K 下的 SSE 与平均轮廓系数")
    print(f"{'聚类数(K)':>8} {'SSE':>12} {'平均轮廓系数':>12}")
    for k, s, sc in zip(ks, sse, scores):
        print(f"{k:>8} {s:>12.1f} {sc:>12.3f}")

    # 肘部图
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sse, "o-")
    plt.xlabel("K"); plt.ylabel("SSE (簇内误差平方和)")
    plt.title("图4-2 K-SSE 曲线（手肘法）")
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "elbow_curve.png"); plt.savefig(p1, dpi=120); plt.close()

    # 轮廓系数图
    plt.figure(figsize=(6, 4))
    plt.plot(ks, scores, "s-")
    plt.xlabel("K"); plt.ylabel("平均轮廓系数")
    plt.title("不同 K 的平均轮廓系数")
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "silhouette_curve.png"); plt.savefig(p2, dpi=120); plt.close()
    print(f"[clustering] 肘部图 -> {p1}\n[clustering] 轮廓系数图 -> {p2}")
    return sse, scores


def fit_best_k(df: pd.DataFrame, features):
    X = df[features].values
    km = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
    df["Cluster_Label"] = km.fit_predict(X)
    print(f"[clustering] 已完成 K={BEST_K} 聚类，簇中心(标准化空间)：")
    print(pd.DataFrame(km.cluster_centers_, columns=features))
    return df, km


def radar_chart(df: pd.DataFrame, features):
    # 把标准化特征按簇求均值，画雷达图（论文图4-3）
    prof = df.groupby("Cluster_Label")[features].mean()
    labels = [f.replace("_", " ") for f in features]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for i in prof.index:
        vals = prof.loc[i].tolist(); vals += vals[:1]
        ax.plot(angles, vals, label=f"群体 {i}")
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_title("图4-3 用户群体雷达图（簇内特征均值）")
    ax.legend(loc="upper right")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "radar_chart.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"[clustering] 雷达图 -> {p}")
    return prof


def boxplot_age(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Cluster_Label", y="Age")
    plt.title("图4-4 各群体年龄分布对比"); plt.xlabel("群体"); plt.ylabel("Age")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "age_boxplot.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"[clustering] 年龄箱线图 -> {p}")


def gender_ratio(df: pd.DataFrame):
    ct = pd.crosstab(df["Cluster_Label"], df["Gender"], normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(6, 4))
    plt.title("图4-5 各群体性别占比对比"); plt.ylabel("比例"); plt.xlabel("群体")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "gender_ratio.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"[clustering] 性别占比图 -> {p}\n{ct.round(3).to_string()}")


def category_dist(df: pd.DataFrame):
    ct = pd.crosstab(df["Cluster_Label"], df["Category"], normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("图4-7 各群体商品类别购买分布对比"); plt.ylabel("比例"); plt.xlabel("群体")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "category_dist.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"[clustering] 商品类别分布图 -> {p}")


def subscription_rate(df: pd.DataFrame):
    rate = df.groupby("Cluster_Label")["Subscription"].apply(lambda s: (s == "Yes").mean())
    rate.plot(kind="bar", figsize=(6, 4), color="teal")
    plt.title("图4-10 各群体订阅率对比"); plt.ylabel("订阅率"); plt.xlabel("群体")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "subscription_rate.png"); plt.savefig(p, dpi=120); plt.close()
    print(f"[clustering] 订阅率图 -> {p}\n{rate.round(3).to_string()}")


def run(df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(FIG_DIR, exist_ok=True)
    elbow_and_silhouette(df, CLUSTER_FEATURES_SCALED)
    df, _ = fit_best_k(df, CLUSTER_FEATURES_SCALED)
    radar_chart(df, CLUSTER_FEATURES_SCALED)
    boxplot_age(df)
    gender_ratio(df)
    category_dist(df)
    subscription_rate(df)
    return df


if __name__ == "__main__":
    from preprocess import run as pre
    df = pre()
    run(df)
