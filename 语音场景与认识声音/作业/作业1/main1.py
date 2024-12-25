# 以下以 a 代表振幅，f 代表频率，v 代表速度，请制作满足要求的动画
# 绘制一个 a=3 f=4 v=0.3 向右移动的正弦波
# 绘制一个 a=5 f=2 v=0.1 向左移动的余弦波
import matplotlib.pyplot as plt
import numpy as np


def sin(a, f, t, theta=0.):
    return a * np.sin(2 * np.pi * f * t + theta)


def cos(a, f, t, theta=0.):
    return a * np.cos(2 * np.pi * f * t + theta)


t = np.linspace(0, 2, 1000)

fig, ax = plt.subplots(2)

line1, = ax[0].plot(t, sin(3, 4, t))
line2, = ax[1].plot(t, cos(5, 2, t))

theta1 = 0.
theta2 = 0.

v1 = -0.3
v2 = 0.1

while True:
    plt.pause(0.0167)

    # 根据速度更新图像
    theta1 += v1
    theta2 += v2
    # 计算新的波形
    wave1 = sin(3, 4, t, theta1)
    wave2 = cos(5, 2, t, theta2)

    # 更新图像
    line1.set_ydata(wave1)
    line2.set_ydata(wave2)

    fig.canvas.draw()

plt.show()
