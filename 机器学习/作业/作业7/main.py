# 使用随机森林来预测学生的分数，并预测是否及格
#
# 具体要求如下:
#
# - 随机 500 个学生，每个学生包含 5 个特征:
#     1. 勤劳
#     2. 勇敢
#     3. 善良
#     4. 正直
#     5. 乐观
# - 每个特征取值为 0~1
# - 随机 500 个分数标签作为成绩，分数取值为 0~100
# - 根据随机分数，以60分为及格线，得到一个是否及格的分类标签，0 代表不及格，1 代表及格
# - 训练一个包含 20 个树的随机森林
# - 使用随机森林预测 分数 和 是否及格
import random

import numpy as np

from tree import DecisionTree


class ForestTree(DecisionTree):
    def __init__(self, sample_idx, feature_idx, max_depth=None):
        super().__init__(max_depth=max_depth)
        self.sample_idx = sample_idx
        self.feature_idx = feature_idx


class RandomForest:
    def __init__(self, n_tree, n_sample, n_feature, max_depth=None):
        self.n_tree = n_tree
        self.n_sample = n_sample
        self.n_feature = n_feature
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        N, D = X.shape
        for i in range(self.n_tree):
            sample_idx = random.sample(range(N), self.n_sample)
            feature_idx = random.sample(range(D), self.n_feature)
            samples = X[sample_idx]
            samples = samples[:, feature_idx]
            labels = y[sample_idx]
            tree = ForestTree(sample_idx, feature_idx, max_depth=self.max_depth)
            tree.fit(samples, labels)
            self.trees.append(tree)

    def predict(self, X, mode='classify'):
        result = []
        for tree in self.trees:
            samples = X[:, tree.feature_idx]
            result.append(tree.predict(samples))
        result = np.array(result).T
        if mode == 'classify':
            return np.array([np.argmax(np.bincount(result[i])) for i in range(result.shape[0])])
        else:
            return np.mean(result, axis=1)


def random_samples(num, n_feature):
    samples = np.random.rand(num, n_feature)
    scores = np.random.randint(0, 101, num)
    # 是否及格的标签
    # 及格的索引
    pass_idx = scores >= 60
    not_pass_idx = scores < 60
    labels = np.empty_like(scores)
    labels[pass_idx] = 1
    labels[not_pass_idx] = 0
    return samples, scores, labels


samples, scores, labels = random_samples(500, 5)

forest1 = RandomForest(20, 400, 4)
forest2 = RandomForest(20, 400, 4)

forest1.fit(samples, scores)  # 第一个森林学习如何预测分数
forest2.fit(samples, labels)  # 第二个森林学习如何预测分类

# 获取测试样本
sample_idx = random.sample(range(500), 10)
test_samples = samples[sample_idx]
test_scores = scores[sample_idx]
test_labels = labels[sample_idx]

print('预测分数')
print(forest1.predict(test_samples, mode='regression'))
print(test_scores)
print('预测分类')
print(forest2.predict(test_samples, mode='classify'))
print(test_labels)
