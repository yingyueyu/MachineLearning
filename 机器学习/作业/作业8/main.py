import random

import numpy as np


# 高斯朴素贝叶斯分类器
class GaussianNaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        # 计算先验
        self.prior = np.bincount(y) / y.shape[0]

        # 分类数
        self.c_values = np.unique(y)
        self.n_c = len(self.c_values)

        self.n_feature = X.shape[1]

        self.means_and_vars = []

        # 分类计算均值和方差
        for i in range(self.n_c):
            c = self.c_values[i]
            samples = X[y == c]
            # 计算每种分类下所有特征的均值和方差
            self.means_and_vars.append([np.mean(samples, axis=0), np.var(samples, axis=0)])
        self.means_and_vars = np.array(self.means_and_vars)

    def predict(self, X):
        # 计算测试样本的条件概率
        lh = self.compute_likelihood(X)
        prior = np.expand_dims(self.prior, 0).repeat(X.shape[0], 0)
        posterior = lh * prior
        return np.argmax(posterior, axis=1)

    def compute_likelihood(self, xi):
        # 计算不同分类下的似然值
        mean = self.means_and_vars[:, 0]
        var = self.means_and_vars[:, 1]
        mean = np.expand_dims(mean, 0).repeat(xi.shape[0], 0)
        var = np.expand_dims(var, 0).repeat(xi.shape[0], 0)
        xi = np.expand_dims(xi, 1).repeat(self.n_c, 1)
        probs = (1 / np.sqrt(2 * np.pi * var)) * np.exp(- ((xi - mean) ** 2 / (2 * var)))
        # 计算所有特征的乘积
        # lh = np.ones((xi.shape[0], self.n_c))
        # for feature in range(self.n_feature):
        #     lh *= probs[:, :, feature]
        lh = np.prod(probs, axis=2)
        return lh


def random_samples(n_sample, n_feature):
    return np.random.randn(n_sample, n_feature)


n = 50
n_sample = 10

X = random_samples(n, 5)
y = np.random.randint(0, 101, n)
y[y <= 60] = 0
y[y > 60] = 1
print(y)

model = NaiveBayes()
model.fit(X, y)

idx = random.sample(range(n), n_sample)
print(model.predict(X[idx]))
print(y[idx])
