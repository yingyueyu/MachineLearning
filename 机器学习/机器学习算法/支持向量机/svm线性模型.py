# 优化问题:
# 1. 找到最小的 1/2 ||w||^2 (二分之一w的范数的平方)
# 2. 约束条件 y_i * ( w^T * x_i + b ) >= 1

# 优化思路:
# 此处简化优化问题，我们让不满足约束条件的样本产生损失
# 不满足约束条件的点，满足 y_i * ( w^T * x_i + b ) < 1
# 这意味着这个点，进入到了间隔带里面，我们希望这个点离间隔带越远越好
# 所以我们已这个点离 1 的差值作为其损失，即 loss = 1 - y_i * ( w^T * x_i + b )

# 训练完后，绘制分界线
# 思路: 找到满足 wx + b = 0 中的 2 个 x，既能画出直线
# 这里展开式子 w1x1 + w2x2 + b = 0
# 假设 x1 已知，求 x2: x2 = - (w1x1 + b / w2)
import numpy as np
from matplotlib import pyplot as plt


# 随机样本
def random_sample(num):
    # 求两个分类的数量
    n1 = num // 2
    n2 = num - n1

    # 标签
    y = np.ones(num)
    y[n2:] = -1

    # 按照正态分布随机样本
    sample1 = np.random.normal((-2, 2), 0.5, (n1, 2))
    sample2 = np.random.normal((2, -2), 0.5, (n1, 2))

    samples = np.vstack((sample1, sample2))

    return samples, y


class SVM:
    def __init__(self, EPOCH=50, lr=1e-3):
        self.W = np.zeros(2)
        self.b = 0
        self.EPOCH = EPOCH
        self.lr = lr

    # 优化问题:
    # 1. 找到最小的 1/2 ||w||^2 (二分之一w的范数的平方)
    # 2. 约束条件 y_i * ( w^T * x_i + b ) >= 1
    def fit(self, X, y):
        fig, ax = plt.subplots()

        # 画样本点
        sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

        # 画决策平面
        x1, x2 = self.get_x1_x2(X)
        line, = ax.plot(x1, x2, 'y--')

        for epoch in range(self.EPOCH):
            # 循环每个样本点
            for i in range(X.shape[0]):
                # 清空梯度

                # 获取一个样本
                xi = X[i]
                yi = y[i]
                # 模型预测
                _y = self.predict(xi)

                # 计算损失
                # 优化思路:
                # 此处简化优化问题，我们让不满足约束条件的样本产生损失
                # 不满足约束条件的点，满足 y_i * ( w^T * x_i + b ) < 1
                # 这意味着这个点，进入到了间隔带里面，我们希望这个点离间隔带越远越好
                # 所以我们已这个点离 1 的差值作为其损失，即 loss = 1 - y_i * ( w^T * x_i + b )
                if yi * _y < 1:
                    # 满足条件的样本，产生损失
                    loss = 1 - yi * _y

                    # 反向传播 计算梯度
                    dw_to_dloss = - yi * xi
                    db_to_dloss = - yi

                    # 优化参数
                    self.W -= self.lr * dw_to_dloss
                    self.b -= self.lr * db_to_dloss

                    # 更新动画
                    plt.pause(0.0167)
                    x1, x2 = self.get_x1_x2(X)
                    line.set_data(x1, x2)
                    fig.canvas.draw()

        plt.colorbar(sc)
        plt.show()

    # 模型预测
    def predict(self, X):
        if len(X.shape) == 1:
            return self.W.T @ X + self.b
        else:
            # 当 X 形状为二维时，例如 (100, 2)
            # 使用张量广播先执行乘法操作
            return np.sum(self.W * X, axis=1) + self.b

    # 训练完后，绘制分界线
    # 思路: 找到满足 wx + b = 0 中的 2 个 x，既能画出直线
    # 这里展开式子 w1x1 + w2x2 + b = 0
    # 假设 x1 已知，求 x2: x2 = - (w1x1 + b / w2)
    def get_x1_x2(self, X):
        x1 = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
        if self.W[0] == 0:
            self.W[0] = 1e-5
        if self.W[1] == 0:
            self.W[1] = 1e-5
        x2 = - (self.W[0] * x1 + self.b) / self.W[1]
        return x1, x2


X, y = random_sample(50)

model = SVM()
model.fit(X, y)
print(model.predict(X))
print(y)
x1, x2 = model.get_x1_x2(X)

# fig, ax = plt.subplots()
#
# sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
# ax.plot(x1, x2, 'y--')
#
# plt.colorbar(sc)
# plt.show()
