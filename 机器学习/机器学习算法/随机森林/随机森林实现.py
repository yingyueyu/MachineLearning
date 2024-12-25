import random

import numpy as np

from tree import DecisionTree

np.random.seed(0)


class ForestTree(DecisionTree):
    # sample_idx: 样本的索引
    # feature_idx: 特征的索引
    # max_depth: 树的最大深度
    def __init__(self, sample_idx, feature_idx, max_depth=None):
        super().__init__(max_depth=max_depth)
        self.sample_idx = sample_idx
        self.feature_idx = feature_idx


class RandomForest:
    # n_tree: 森林中树的个数
    # n_samples: 每棵树训练时样本的个数
    # n_features: 每棵树训练时特征选择的个数
    def __init__(self, n_tree, n_samples, n_features, max_depth=None):
        self.n_tree = n_tree
        self.n_samples = n_samples
        self.n_features = n_features
        self.max_depth = max_depth
        # 保存森林中的所有树
        self.trees = []

    def fit(self, X, y):
        # N: 样本数
        # D: 特征数
        N, D = X.shape
        # 循环构造树
        for i in range(self.n_tree):
            # 对样本进行采样
            sample_idx = random.sample(range(N), self.n_samples)
            feature_idx = random.sample(range(D), self.n_features)
            # 获取样本
            sample = X[sample_idx]
            sample = sample[:, feature_idx]
            # 获取标签
            label = y[sample_idx]
            # 构造树
            tree = ForestTree(sample_idx, feature_idx, self.max_depth)
            tree.fit(sample, label)
            self.trees.append(tree)

    # mode: 可选项 classify 或 regression，classify 时，取每棵树统计结果的最大值，regression 时取平均值
    def predict(self, X, mode='classify'):
        result = []
        for tree in self.trees:
            # 获取当前树要预测的特征
            samples = X[:, tree.feature_idx]
            # 预测结果
            result.append(tree.predict(samples))
        result = np.array(result).T
        print(result)
        print(result.shape)

        if mode == 'classify':
            return np.array([np.argmax(np.bincount(result[i])) for i in range(result.shape[0])])
        else:
            return np.mean(result, axis=1)


X = np.random.randint(0, 3, (100, 5))
y = np.random.randint(0, 2, 100)

forest = RandomForest(10, 50, 4, 10)
forest.fit(X, y)

result = forest.predict(X, mode='regression')
print(result)
print(result.shape)
