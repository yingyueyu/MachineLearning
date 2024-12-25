# 高斯朴素贝叶斯: 用于特征满足高斯分布的情况，来进行分类
import random

import numpy as np
from matplotlib import pyplot as plt


class GaussianNB:
    def fit(self, X, y):
        # 计算先验概率
        self.prior = np.bincount(y) / len(y)

        # 不同分类的值
        self.c_value = np.unique(y)

        # 分类个数
        self.n_c = len(self.c_value)

        # 特征个数
        self.n_feature = X.shape[1]

        # 所有类别下所有特征的均值和方差
        self.means_and_vars = []

        # 计算不同分类下所有特征的均值和方差
        for i in range(self.n_c):
            # 取出一个分类
            c = self.c_value[i]
            # 取出该分类的样本
            samples = X[y == c]
            self.means_and_vars.append([samples.mean(axis=0), samples.var(axis=0)])

        self.means_and_vars = np.array(self.means_and_vars)

    def predict(self, X):
        likelihood = self.compute_likelihood(X)
        # 求后验概率
        posterior = self.prior * likelihood
        # 求最大后验概率
        return np.argmax(posterior, axis=1)


    def compute_likelihood(self, xi):
        # 获取均值和方差
        mean = self.means_and_vars[:, 0]
        var = self.means_and_vars[:, 1]
        # 变形处理
        xi = np.expand_dims(xi, 1).repeat(self.n_c, 1)
        mean = np.expand_dims(mean, 0).repeat(xi.shape[0], 0)
        var = np.expand_dims(var, 0).repeat(xi.shape[0], 0)
        prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(- ((xi - mean) ** 2 / (2 * var)))
        # 求所有特征概率的乘积
        # likelihood = np.ones((xi.shape[0], self.n_c))
        # for feature in range(self.n_feature):
        #     likelihood *= prob[:, :, feature]
        likelihood = prob.prod(axis=2)
        return likelihood


X = np.random.normal(0, 2, (100, 5))
y = np.random.randint(0, 2, 100)

model = GaussianNB()
model.fit(X, y)

idx = random.sample(range(100), 10)
print(model.predict(X[idx]))
print(y[idx])
