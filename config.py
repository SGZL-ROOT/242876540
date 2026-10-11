"""
配置文件 —— 对应论文《基于机器学习的电商用户行为和销售分析》
集中存放路径、列名、模型超参，方便复现时统一修改。

注意（重要）：
论文正文说用的是「阿里天池 淘宝用户行为数据集」（字段只有 user_id / item_id /
behavior_type / timestamp），但附录代码实际读取的 CSV 列是 Age、PurchaseAmount(USD)、
ReviewRating、Previous_Purchases、Gender、Category、Color、PaymentMethod、Subscription 等。
这两套字段并不兼容——淘宝行为数据里根本没有「年龄/评分/历史购买次数」这些列。
因此本复现工程按「附录代码实际需要的字段」来组织数据，并内置一个合成数据生成器，
让你不依赖外部数据集也能跑通整条流水线；同时 README 里给出了接入真实数据集的映射。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figs")
RAW_CSV = os.path.join(DATA_DIR, "ecommerce_data.csv")

# 聚类用「核心数值特征」（与论文附录代码一致：4 个标准化/归一化后的数值特征）
CLUSTER_NUMERIC_FEATURES = ["Age", "PurchaseAmount(USD)", "ReviewRating", "Previous_Purchases"]
# 聚类核心特征对应的变换后列名
CLUSTER_FEATURES_SCALED = [
    "Age_Scaled",
    "Purchase_Amount_Scaled",
    "Review_Rating_Norm",
    "Previous_Purchases_Norm",
]
# 聚类后用于「画像分析」的辅助类别特征（不参与距离计算）
PROFILE_CATEGORICAL = ["Gender", "Category", "Color", "PaymentMethod", "Subscription", "PromoCode"]

# 缺失值 / 异常值处理参数
IQR_K = 1.5  # 盖帽法上下界倍数

# K-Means 相关
KMIN, KMAX = 2, 8
BEST_K = 3
RANDOM_STATE = 42

# 销售预测相关
WINDOW = 7            # 滑动窗口 T=7（论文 5.1.1）
TEST_RATIO = 0.2
LSTM_UNITS = 64
LSTM_EPOCHS = 150
LSTM_BATCH = 32
LSTM_DROPOUT = 0.1

N_TRANSACTIONS = 100_000  # 与论文「抽取 10 万条记录」一致

# 中文显示：尝试设置支持中文的字体，避免图表标题出现方块（□）。
# 若当前环境无中文字体则静默跳过，仅影响显示、不影响计算。
try:
    import matplotlib
    _cjk = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "WenQuanYi Micro Hei"]
    for _f in _cjk:
        try:
            matplotlib.rcParams["font.family"] = [_f]
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

