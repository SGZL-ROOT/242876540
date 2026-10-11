# 电商用户行为与销售分析 —— 论文实验复现

复现对象：王钊《基于机器学习的电商用户行为和销售分析》（郑州财经学院，2026）的课程/学士学位论文实验。

本工程把论文里分散在正文与附录的代码，整理成一条**可一键运行**的流水线，覆盖论文三大实验：
第二章理论基础之外的可复现部分，即：
- 第三章 数据处理与特征选择（缺失值 / 异常值 / 标准化归一化 / 相关性 / 高基数编码）
- 第四章 用户画像构建（K-Means 聚类、手肘法 + 轮廓系数选 K、雷达图/箱线图画像）
- 第五章 销售预测（ARIMA / XGBoost / LSTM 对比，RMSE/MAE/R²）

---

## ⚠️ 最重要的一个坑：论文「描述的」数据集 ≠ 「代码实际用的」数据集

- 正文说：用「阿里天池 淘宝用户行为数据集」，字段只有 `user_id / item_id / behavior_type(点击/收藏/加购/购买) / timestamp`。
- 附录代码（以及第 4、5 章的特征：年龄、性别、购买金额、评价评分、历史购买次数、商品类别、颜色、支付方式、订阅率、促销码）却来自**另一套电商交易数据**，其典型列是：
  `Age, PurchaseAmount(USD), ReviewRating, Previous_Purchases, Gender, Category, Color, PaymentMethod, Subscription, PromoCode, Timestamp`。

淘宝行为数据里**根本不存在年龄/评分/历史购买次数**这些列，所以「用淘宝数据集跑附录代码」在数学上不成立。这是本科论文里很常见的「叙述一套、实现另一套」的脱节。

**复现结论**：本工程按「附录代码实际需要的字段」组织数据。为了让你不依赖任何外部数据就能跑通，内置了 `generate_data.py` 合成一份字段对齐的数据（并刻意埋入 3 类客户分群，使 K-Means 能复现出 K=3、轮廓系数≈0.6 的效果）。

### 接入真实数据集
把你的 CSV 放到 `data/ecommerce_data.csv`，列名对齐 `generate_data.py` 里的 `OUTPUT_COLUMNS` 即可，其余代码无需改动：
```
TransactionID, CustomerID, Gender, Age, Category, Color, PaymentMethod,
Subscription, PromoCode, PurchaseAmount(USD), ReviewRating, Previous_Purchases, Timestamp
```
若你坚持用「淘宝用户行为数据集」：需要先做大量特征工程（从行为日志里推导出每个用户的年龄、性别、评分、历史购买次数、品类/颜色/支付方式偏好等），这些字段在原数据里不存在，**论文并未给出这部分推导代码**，属于需要你自己补全的缺口。

---

## 已验证的运行结果（本工程合成数据，固定随机种子）

在 Python 3.13 + venv 中已完整跑通（全流程约 30~40 秒，不含 tensorflow 安装）。

**聚类（第四章，表4-1 对应）**
| 聚类数 K | SSE | 平均轮廓系数 |
|---|---|---|
| 2 | 77890.9 | 0.561 |
| **3** | **18628.6** | **0.692（峰值）** |
| 4 | 15489.0 | 0.544 |
| 5 | 13010.3 | 0.398 |

K=3 既是 SSE 肘部，又是轮廓系数峰值 → 与论文「K=3 最优」一致。输出图：`elbow_curve.png`、`silhouette_curve.png`、`radar_chart.png`、`age_boxplot.png`、`gender_ratio.png`、`category_dist.png`、`subscription_rate.png`。

**销售预测（第五章，表5-1 对应）**
| 算法模型 | RMSE | MAE | R² |
|---|---|---|---|
| ARIMA（传统时序） | 852.9 | 345.1 | -0.013 |
| XGBoost（集成学习） | 124.8 | 94.0 | 0.978 |
| LSTM（深度学习） | 127.5 | 97.7 | 0.977 |

核心结论复现：传统 ARIMA 因无法利用「大促/周末」等外部特征、且处理不规则尖峰能力差，远差于引入行为/事件特征的 XGBoost 与 LSTM；二者在该合成数据上接近上限（真实数据中 LSTM 对非线性波动的优势会进一步拉开差距，对应论文 LSTM 最优）。

> 注意：以上数字来自合成数据，仅用于验证 Pipeline 与模型排序；接入真实数据集后具体数值会不同，但方法框架一致。

---

## 运行方式

```bash
# 1. 建议用虚拟环境
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 一键复现（自动生成合成数据 → 预处理 → 聚类 → 预测）
python main.py
```

不装 `tensorflow` 也能跑（会跳过 LSTM 并提示）；装了则 LSTM 也参与对比。

---

## 目录结构
```
ecommerce_ml_repro/
├── config.py          # 路径、列名、超参集中配置
├── generate_data.py   # 合成数据（字段对齐附录代码）
├── preprocess.py      # 第三章：缺失值/IQR盖帽/标准化归一化/相关性/频次编码
├── clustering.py      # 第四章：肘部法+轮廓系数、K=3、雷达图/箱线图/画像
├── forecast.py        # 第五章：ARIMA/XGBoost/LSTM，输出 RMSE/MAE/R²
├── main.py            # 串联全流程
├── requirements.txt
├── data/              # 生成的 ecommerce_data.csv、clean_data.csv
└── figs/              # 所有产出图（相关性热力图、肘部图、雷达图、预测对比等）
```

## 与论文结果的差异说明
- **聚类**：合成数据刻意设计成 3 个分群，因此能复现「K=3、轮廓系数较高」的结论与画像趋势；真实数据接入后具体数值会变。
- **销售预测**：论文表 5-1（ARIMA 452.3 / XGBoost 185.6 / LSTM 142.1）是**其原始数据**上的结果。本工程复现的是同一套 Pipeline 与**模型排序（LSTM 最优）**，具体 RMSE/MAE/R² 数值取决于你喂入的数据，不会也不应该和论文完全一致。

## 已修正的论文附录代码 bug
1. 附录用 `kmeans.fit_transform(features)` 当作簇标签——错误（它返回的是到各中心的距离矩阵）。正确做法：`fit_predict`。
2. 附录只演示了单列的标准化/归一化；本工程批量处理全部核心数值特征并保留原列。
3. LSTM 反归一化时只还原目标列（销量），避免特征被错误反变换。
