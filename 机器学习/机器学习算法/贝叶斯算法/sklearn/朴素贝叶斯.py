import numpy as np
from sklearn.naive_bayes import CategoricalNB

# 官方 api: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#gaussiannb

# 朴素贝叶斯有:
# 1. GaussianNB 高斯朴素贝叶斯: 适用于特征是连续值的情况
# 2. MultinomialNB 多项式朴素贝叶斯: 适用于特征是离散值的情况
# 3. BernoulliNB 伯努利朴素贝叶斯: 适合二元特征
# 4. ComplementNB 补全朴素贝叶斯: 是 MultinomialNB 的变体，用于处理不平衡数据集（有些分类的数据特别少，有些又特别多）
# 5. CategoricalNB 类别朴素贝叶斯: 适合特征属于多种类别值得情况，例如，例子中我们按天气，气温，适度，风力等分类，判断是否适合打篮球

X = np.array([
    [0, 2, 1, 0],  # 晴, 高, 高, 弱
    [0, 2, 1, 1],  # 晴, 高, 高, 强
    [1, 2, 1, 0],  # 阴, 高, 高, 弱
    [2, 1, 1, 0],  # 雨, 中, 高, 弱
    [2, 0, 0, 0],  # 雨, 低, 正常, 弱
    [2, 0, 0, 1],  # 雨, 低, 正常, 强
    [1, 0, 0, 1],  # 阴, 低, 正常, 强
    [0, 1, 1, 0],  # 晴, 中, 高, 弱
    [0, 0, 0, 0],  # 晴, 低, 正常, 弱
    [2, 1, 0, 0],  # 雨, 中, 正常, 弱
])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])

# 参数:
# alpha: 平滑参数，防止概率为0。
#   假设样本为 [雨, 中, 正常, 弱]
#   则，P(适合打篮球) = P(雨|适合) * P(中|适合) * P(正常|适合) * P(弱|适合)
#   如果，P(雨|适合) = 0，则整个概率为0，所以需要平滑
# force_alpha: 若为 False，则当 alpha 小于 1e-10 时，设置为 1e-10。否则 alpha 保持不变
# fit_prior: 是否学习先验概率，若为 False，则使用均匀分布。默认为 True
# class_prior: 先验概率，若为 None，则使用 fit_prior 参数。否则，使用 class_prior 参数
# min_categories: 每个特征的最小分类数量，例如天气特征有三种类，则写 3，或4中特征的分类分别是 [3, 2, 2, 2]
model = CategoricalNB(
    alpha=1,
    force_alpha=False,
    fit_prior=True,
    # class_prior=[0.2, 0.8],
    min_categories=[3, 3, 2, 2]
)

model.fit(X, y)

print(model.predict(X))
print(y)
