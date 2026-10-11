"""
main.py —— 一键复现论文实验
流程：生成合成数据 -> 第三章预处理 -> 第四章聚类画像 -> 第五章销售预测
用法：python main.py
"""

import os
import generate_data
import preprocess
import clustering
import forecast


def main():
    print("=" * 60)
    print("电商用户行为与销售分析 —— 论文实验复现")
    print("=" * 60)
    if not os.path.exists(generate_data.RAW_CSV):
        print("\n[1/4] 生成合成数据 ...")
        generate_data.main()
    else:
        print("\n[1/4] 检测到已存在 data/ecommerce_data.csv，跳过生成")

    print("\n[2/4] 第三章 数据预处理与特征选择 ...")
    df = preprocess.run()

    print("\n[3/4] 第四章 用户画像构建（K-Means 聚类）...")
    df = clustering.run(df)

    print("\n[4/4] 第五章 销售预测模型对比 ...")
    forecast.run()

    print("\n全部完成。图表位于 figs/，清洗后数据位于 data/clean_data.csv。")


if __name__ == "__main__":
    main()
