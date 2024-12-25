import torch
import matplotlib.pyplot as plt
from random import random

fig = plt.figure()
w1 = torch.arange(-1, 1, 0.01)
w2 = torch.arange(-1, 1, 0.01)

ax1 = fig.add_subplot(111)
x, y = torch.meshgrid((w1, w2), indexing='ij')
loss = x ** 2 + 2 * y ** 2
ax1.contour(x, y, loss)

# 随机梯度下降法  SGD
epochs = 20
points = []
w1_pre = 1
w2_pre = 0.8
v_t_w1 = 0
v_t_w2 = 0
for epoch in range(epochs):
    e = w2_pre ** 2 + 2 * w2_pre ** 2
    points.append([w1_pre, w2_pre])
    # 动量法
    v_t_w1 += 2 * w1_pre
    v_t_w2 += 4 * w2_pre
    w1_pre -= 0.01 * v_t_w1
    w2_pre -= 0.01 * v_t_w2

points = torch.tensor(points)
ax1.plot(points[:, 0], points[:, 1], 'ro-')

plt.show()
