# 目的: 找到一条直线，去拟合我们的真实数据
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# 训练步骤:
# 1. 清空梯度
# 2. 模型预测
# 3. 计算损失值
# 4. 计算梯度
# 5. 更新参数


# opencv 视频后端  ffmpeg
# matplotlib 画图后端  TkAgg PyQT
# backend 后端
# 指定后端
# matplotlib.use('TkAgg')


# 超参数
lr = 0.01


# 造一组真实数据

# 期望函数
def expect_fn(x):
    return 1.2 * x + -10


# 横坐标
x = np.arange(10)
y = expect_fn(x)
# 加入噪声
y += np.random.normal(-2, 2, 10)

# 声明参数
w = 0
b = 0


# 声明模型
def model(x):
    return w * x + b


# 均方误差损失函数
def MSELoss(_y, y):
    return np.sum((_y - y) ** 2, axis=0)


# 显示样本
fig, ax = plt.subplots()
# 画点
ax.scatter(x, y, 64, c='r')
# 画期望直线
ax.plot(x, expect_fn(x), 'y--')

# 画模型预测的结果
line, = ax.plot(x, model(x), c='b')

# 训练
for i in range(1000):
    # 1. 清空梯度
    grad = None
    # 2. 模型预测
    _y = model(x)

    # 更新模型预测的直线
    line.set_ydata(_y)
    fig.canvas.draw()
    plt.pause(0.0167)

    # 3. 计算损失值
    loss = MSELoss(_y, y)
    print(f'loss: {loss}')
    # 4. 计算梯度
    dy_to_dloss = 2 * (_y - y)
    dw_to_dy = x
    # w 的导数
    grad_w = (dy_to_dloss * dw_to_dy).mean()
    # b 的导数
    db_to_dy = 1
    grad_b = (dy_to_dloss * db_to_dy).mean()
    # 5. 更新参数
    w = w - lr * grad_w
    b = b - lr * grad_b

# 画图
plt.show()
