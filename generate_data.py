"""
generate_data.py —— 合成一份「与论文附录代码字段一致」的电商交易数据。

为什么需要它：论文正文描述的淘宝行为数据集与附录代码字段不匹配（见 config.py 注释）。
为了让复现代码「开箱即跑」，这里生成一份结构相同的合成数据，并且刻意埋入 3 类客户分群，
使得 K-Means 能复现出论文里 K=3、轮廓系数≈0.6+ 的效果。

真实数据接入方式：把你的 CSV 改名/放到 data/ecommerce_data.csv，列名对齐本脚本输出的列即可，
无需改其它代码。列名见 OUTPUT_COLUMNS。
"""

import os
import numpy as np
import pandas as pd
from config import RAW_CSV, N_TRANSACTIONS, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

OUTPUT_COLUMNS = [
    "TransactionID", "CustomerID", "Gender", "Age",
    "Category", "Color", "PaymentMethod", "Subscription", "PromoCode",
    "PurchaseAmount(USD)", "ReviewRating", "Previous_Purchases", "Timestamp",
]

# 三个潜在客户分群（对应论文里的高价值忠诚型 / 沉默大众型 / 年轻高消费型）
# 每个分群给出各特征的均值，生成时加高斯噪声，使 K-Means 能自然恢复出 3 类。
SEGMENTS = {
    "loyal": {     # 群体0：年长、高频、低客单、高评分（高价值忠诚型）
        "w": 0.40, "Age": (55, 4), "Prev": (25, 3), "Amt": (40, 9), "Rate": (4.6, 0.25),
        "gender": "Male",
    },
    "silent": {    # 群体1：中年、中频、中客单、中评分（沉默大众型）
        "w": 0.40, "Age": (35, 4), "Prev": (8, 2), "Amt": (65, 11), "Rate": (3.4, 0.3),
        "gender": "Male",
    },
    "young": {     # 群体2：年轻、低频、高客单、低评分（年轻高消费型）
        "w": 0.20, "Age": (23, 3), "Prev": (3, 1.5), "Amt": (160, 28), "Rate": (2.3, 0.35),
        "gender": "Female",
    },
}

CATEGORIES = ["Electronics", "Clothing", "Home", "Sports", "Books", "Beauty"]
COLORS = ["Red", "Blue", "Green", "Black", "White", "Yellow"]
PAYMENTS = ["Credit Card", "PayPal", "Debit Card", "Cash"]


def make_customers(n_customers: int) -> pd.DataFrame:
    rows = []
    cid = 0
    for seg, p in SEGMENTS.items():
        k = int(round(n_customers * p["w"]))
        for _ in range(k):
            cid += 1
            age = int(np.clip(rng.normal(*p["Age"]), 18, 70))
            prev = int(np.clip(rng.normal(*p["Prev"]), 0, 60))
            amt = float(np.clip(rng.normal(*p["Amt"]), 5, 500))
            rate = float(np.clip(rng.normal(*p["Rate"]), 1, 5))
            # 性别以分群为主但有少量噪声
            gender = p["gender"] if rng.random() < 0.85 else ("Female" if p["gender"] == "Male" else "Male")
            rows.append({
                "CustomerID": f"C{cid:06d}",
                "Gender": gender,
                "Age": age,
                "Previous_Purchases": prev,
                "PurchaseAmount(USD)": round(amt, 2),
                "ReviewRating": round(rate, 2),
                "Segment": seg,
            })
    return pd.DataFrame(rows)


def main():
    # 客户数取交易数的约 1/8，保证同一客户多次出现
    n_customers = max(2000, N_TRANSACTIONS // 8)
    customers = make_customers(n_customers)

    # 生成交易记录
    cust_sample = customers.sample(n=N_TRANSACTIONS, replace=True, random_state=RANDOM_STATE)
    df = cust_sample.reset_index(drop=True)
    df["TransactionID"] = np.arange(1, N_TRANSACTIONS + 1)
    # 交易金额/评分在客户基线附近再加一点扰动（保持合成分群在聚类特征上可分）
    df["PurchaseAmount(USD)"] = (df["PurchaseAmount(USD)"] * rng.normal(1.0, 0.08, N_TRANSACTIONS)).clip(5, 600).round(2)
    df["ReviewRating"] = (df["ReviewRating"] + rng.normal(0, 0.15, N_TRANSACTIONS)).clip(1, 5).round(2)
    df["Category"] = rng.choice(CATEGORIES, N_TRANSACTIONS)
    df["Color"] = rng.choice(COLORS, N_TRANSACTIONS)
    df["PaymentMethod"] = rng.choice(PAYMENTS, N_TRANSACTIONS, p=[0.4, 0.3, 0.2, 0.1])
    df["Subscription"] = np.where(rng.random(N_TRANSACTIONS) < 0.3, "Yes", "No")
    df["PromoCode"] = np.where(rng.random(N_TRANSACTIONS) < 0.4, "Yes", "No")
    # 时间戳：覆盖 ~2 年，产生日度序列用于预测
    start = pd.Timestamp("2023-01-01")
    days = rng.integers(0, 730, N_TRANSACTIONS)
    secs = rng.integers(0, 86400, N_TRANSACTIONS)
    df["Timestamp"] = [start + pd.Timedelta(days=int(d), seconds=int(s)) for d, s in zip(days, secs)]

    df = df[OUTPUT_COLUMNS]
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"[generate_data] 已生成 {len(df)} 条记录 -> {RAW_CSV}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
