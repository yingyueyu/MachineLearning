import random

import numpy as np
from matplotlib import pyplot as plt


# 总结训练步骤:
# 1. 保存参数: X, y, n_sample, n_feature
# 2. 初始化参数: alpha, gamma
# 3. 声明核函数
# 4. 训练循环
#       1. 计算 alpha_i 的梯度
#       2. 根据限制条件计算优化前后的 alpha_i
#       3. 为了满足优化条件2，计算误差 error
#       4. 选择一个非 i 的 j，找到另一个 alpha_j
#       5. 通过 error 计算 alpha_j 的新值
#       6. 优化 alpha_i alpha_j
#       7. alpha 优化结束后，再解 b

# 总结预测步骤:
# 1. 计算公式中求和的前半部分
# 2. 再将求和结果加上 b 得到值 r
# 3. 通过判断 r 的正负，最终得到结果 y


def get_samples():
    X = np.array([
        [0, 2],
        [2, 0],
        [0, -2],
        [-2, 0],
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
    ])

    y = [1, 1, 1, 1, -1, -1, -1, -1]

    return X, y


class SVM:
    def __init__(self, gamma='scale', C=1., EPOCH=100, lr=1e-3):
        self.gamma = gamma  # gamma 配置
        self.gamma_value = None  # gamma 值， gamma = 1 / 2*sigma^2
        self.C = C  # 惩罚系数
        self.EPOCH = EPOCH  # 迭代次数
        self.lr = lr  # 学习率
        self.n_sample = None  # 样本数
        self.n_feature = None  # 特征数
        self.EPOCH = EPOCH
        self.lr = lr
        self.alpha = None  # 拉格朗日乘子
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_sample, self.n_feature = X.shape
        # 初始化 alpha
        self.alpha = np.zeros(self.n_sample)
        # 计算gamma
        if self.gamma.isdigit():
            self.gamma_value = self.gamma
        elif self.gamma == 'scale':
            self.gamma_value = 1 / (self.n_sample * np.mean(np.var(X, axis=0)))
        else:
            ValueError('gamma 值异常')

        # 训练循环
        for epoch in range(self.EPOCH):
            for i in range(self.alpha.shape[0]):
                # 计算梯度
                grad_i = self.compute_grad_i(i)
                # 限制条件:
                # 1. 0 <= a_i <= C
                # 2. a^T @ y = 0
                # 优化 alpha_i
                old_alpha_i = self.alpha[i]
                new_alpha_i = old_alpha_i + self.lr * grad_i
                # 限制条件 1
                new_alpha_i = np.maximum(np.minimum(new_alpha_i, self.C), 0)
                # 计算误差 error
                error = old_alpha_i * self.y[i] - new_alpha_i * self.y[i]
                # 选择另一个alpha_j
                j = self.select_j(i)
                # 计算新的 alpha_j
                new_alpha_j = (self.alpha[j] * self.y[j] - error) / self.y[j]
                # 跟新 alpha
                self.alpha[i] = new_alpha_i
                self.alpha[j] = new_alpha_j

            # 解 b
            b = 0
            count = 0
            for i in range(self.alpha.shape[0]):
                # alpha_i 从 0 ~ C 中选取
                if 0 < self.alpha[i] < self.C:
                    sum = 0
                    for j in range(self.alpha.shape[0]):
                        sum += self.alpha[j] * self.y[i] * self.y[j] * self.kernel(self.X[i], self.X[j])
                    b += (1 - sum) / self.y[i]
                    count += 1
            self.b = b / count

    # x1,x2: 两个样本
    def kernel(self, x1, x2):
        return np.exp(-(self.gamma_value * np.sum((x1 - x2) ** 2)))

    # 计算 alpha_i 的梯度
    def compute_grad_i(self, i):
        sum = 0
        for j in range(self.alpha.shape[0]):
            if i == j:
                sum += self.y[i] * self.y[j] * self.kernel(self.X[i], self.X[j])
            else:
                sum += 0.5 * self.alpha[j] * self.y[i] * self.y[j] * self.kernel(self.X[i], self.X[j])
        return 1 - sum / self.alpha.shape[0]

    def select_j(self, i):
        return random.sample([j for j in range(self.alpha.shape[0]) if j != i], 1)[0]

    # 预测
    def predict(self, X):
        result = []

        # 循环样本的个数
        for j in range(X.shape[0]):
            sum = 0
            for i in range(self.alpha.shape[0]):
                sum += self.alpha[i] * self.y[i] * self.kernel(self.X[i], X[j])
            r = sum + self.b
            result.append(1 if r >= 0 else -1)

        return result


X, y = get_samples()

model = SVM()
model.fit(X, y)
print(model.predict(X))
print(y)

fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

plt.show()
