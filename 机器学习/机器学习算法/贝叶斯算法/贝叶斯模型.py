# 我们已新冠检测为例制作一个贝叶斯模型
# 我们的模型是在离散数据下训练的，并非连续数据
# 连续数据下使用的贝叶斯公式需要使用高斯分布来计算似然值
import random

import numpy as np


# 整体步骤如下:
# 1. 计算初始后验概率
#    1.1 计算先验概率。先验概率可以在当前数据集上计算得到，也可以根据历史上的统计得到
#    1.2 计算似然值。似然值可以通过当前数据集得到，也能通过历史可信统计得到
#    1.3 计算边际概率
#    1.4 计算后验概率
# 2. 接收新证据
#    2.1 计算新证据的似然值。新证据的似然值可以通过当前数据集得到，也能通过历史可信统计得到
#    2.2 计算新证据的边际概率
# 3. 更新后验概率


# 捋清楚已知参数和未知参数:
# 已知参数:
# 1. X: 已知证据
# 2. y: 已知样本标签
# 3. 先验概率: 先验概率是根据已知数据集得到的，也可以是估计的
# 4. 似然值: 似然值是根据已知数据集得到的
# 未知参数:
# 1. 边际概率: 边际概率需要根据似然值和先验概率求得
# 2. 后验概率: 贝叶斯公式的结果


class BayesModel:
    def __init__(self, prior=None, likelihood=None):
        '''
        初始化
        :param prior: 先验概率
        :param likelihood: 似然值
        '''
        self.prior = prior
        self.likelihood = likelihood
        self.posterior = None  # 后验概率
        self.marginal = None  # 边际概率

    def fit(self, X, y, prior=None, likelihood=None):
        '''
        训练模型
        :param X: 已知证据
        :param y: 已知样本标签
        :param prior: 先验概率
        :param likelihood: 似然值
        :return:
        '''
        # 1.1 计算先验概率。先验概率可以在当前数据集上计算得到，也可以根据历史上的统计得到
        # 确保 self.prior 不是空
        if prior is not None:
            self.prior = prior
        elif self.prior is None:
            # 计算先验概率
            tmp = np.bincount(y)
            self.prior = (tmp / tmp.sum()).reshape(-1, 1)

        # 1.2 计算似然值。似然值可以通过当前数据集得到，也能通过历史可信统计得到
        if likelihood is not None:
            self.likelihood = likelihood
        elif self.likelihood is None:
            # 计算似然值
            self.likelihood = self.compute_likelihood(X, y)

        # 1.3 计算边际概率
        self.marginal = np.sum(self.likelihood * self.prior, axis=0)
        # 1.4 计算后验概率
        self.posterior = self.likelihood * self.prior / self.marginal

    # 计算似然值
    def compute_likelihood(self, X, y):
        # 获取可能的 y 值
        A_class = np.unique(y)
        # 获取可能 y 值的个数  2
        n_class = len(A_class)
        # 获取特征个数
        n_feature = X.shape[1]

        likelihood = np.empty((n_class, n_feature))

        for i in range(n_class):
            # 获取事件 A 的值
            A_value = A_class[i]
            # 满足条件 A 的样本
            samples = X[y == A_value]
            # 循环遍历特征
            for feature in range(n_feature):
                # 统计检测为阳和阴的个数
                tmp = np.bincount(samples[:, feature].astype(int))
                # 计算似然值
                lh = tmp / tmp.sum()
                # 保存似然值
                # 取为 1 的值作为该特征的似然值（也就是此处计算的是检测卡阳的概率）
                likelihood[i, feature] = lh[1]

        return likelihood

    # 更新证据
    def update(self, X=None, y=None, likelihood=None):
        assert (X is not None and y is not None) or likelihood is not None

        # 本轮的先验概率是上一轮的后验概率
        self.prior = self.posterior

        # 2.1 计算新证据的似然值。新证据的似然值可以通过当前数据集得到，也能通过历史可信统计得到
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            self.likelihood = self.compute_likelihood(X, y)
        # 2.2 计算新证据的边际概率
        self.marginal = np.sum(self.likelihood * self.prior, axis=0)
        # 更新后验概率
        self.posterior = self.likelihood * self.prior / self.marginal


# 是否感染新冠的样本
# 此处的特征是: 新冠检测卡的结果
# 0: 检测卡结果为阴 1: 检测卡结果为阳
X = np.zeros((100, 1))
# 是否感染新冠的真实值
y = np.zeros(100, dtype=int)

# 随机 10 个人检测卡为阳
sun_idx = random.sample(range(100), 10)
X[sun_idx] = 1

# 从检测结果为阳的 10 个人中，采样 8 个人让其为真阳
true_sun_idx = random.sample(sun_idx, 8)
y[true_sun_idx] = 1

model = BayesModel(prior=np.array([[0.8], [0.2]]))

# 取 80 个人作为第一轮训练样本
idx = random.sample(range(100), 80)
samples = X[idx]
labels = y[idx]

# 第一轮学习
model.fit(samples, labels)
print(model.posterior)

# 补充证据
model.update(X, y)
print(model.posterior)
