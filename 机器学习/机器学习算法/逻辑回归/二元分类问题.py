# 什么是二元分类问题？
# 就是一件事物只有两种可能的结果，比如垃圾邮件和非垃圾邮件，肿瘤是恶性的还是良性的，等等。
# 二元分类问题可以用逻辑回归来解决
# 这里我们已以下例子为例:
# 我们对多个手机样本进行分类，手机被分为遥遥领先机和普通机
import random

import numpy as np
from matplotlib import pyplot as plt

# 随机 10 台手机的特征值，我们用不同的数字代表不同的手机
# 因为 sigmoid 的数学性值，值越大越难收敛，所以这里取 1~2 间的数字
# 若使用 np.arange(10)，则会发现 第一个数字 0 永远不收敛，数字越大越难收敛
# 所以，后面我们使用二元交叉熵损失函数而非均方误差损失函数

x = np.arange(10) + 1
y = np.ones(10)
idx = random.sample(list(np.arange(10)), 5)
y[idx] = 0

# 声明权重参数
w = np.zeros(10)

# 预测

# 建立模型
def model(x):
    # w * x: 线性回归模型
    # sigmoid: 激活
    return sigmoid(w * x)


# 二元分类的激活函数使用 sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BCELoss(_y, y):
    return - np.mean(y * np.log(_y) + (1 - y) * np.log(1 - _y), axis=0)


# BCELoss 的求导函数
def binary_cross_entropy_derivative(_y, y):
    _y[_y == 1] = 0.999999  # 防止除以 0
    return - (y / _y) + ((1 - y) / (1 - _y))


fig, ax = plt.subplots()

ax.scatter(x, y, 64, c='r')

# 绘制模型预测的初始点位
sc = ax.scatter(x, model(x), 48, c='b')

for epoch in range(1000):
    # 1. 清空梯度
    grad = None
    # 2. 模型预测
    _y = model(x)

    # 更新点的纵坐标
    sc.set_offsets(np.c_[x, _y])
    fig.canvas.draw()
    plt.pause(0.0167)

    # 3. 计算损失
    loss = BCELoss(_y, y)
    print(loss)
    # 4. 计算梯度 反向传播
    grad = binary_cross_entropy_derivative(_y, y)
    # 5. 更新参数
    w = w - 0.001 * grad

plt.show()
