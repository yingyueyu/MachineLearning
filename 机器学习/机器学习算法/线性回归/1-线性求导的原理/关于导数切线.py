import numpy as np
import matplotlib.pyplot as plt

# 构建曲线的公式
# 自变量：x  其次是X的区间（0,6）
x = np.linspace(0, 6, 20)

# 因变量：y  满足公式 y = (x - 3) ** 2
# a : 水平方向位移
# b : 垂直方向位移
#
a = 3
b = 2
c = 2


# y1 = x ** 2
# y2 = (x - a) ** 2
# y3 = (x - a) ** 2 + b
# y4 = c * (x - a) ** 2 + b

# plt.subplot(221)
# plt.xlim(0, 6)
# plt.ylim(0, 20)
# plt.plot(x, y1, 'r-')
# plt.subplot(222)
# plt.xlim(0, 6)
# plt.ylim(0, 20)
# plt.plot(x, y2, 'r-')
# plt.subplot(223)
# plt.xlim(0, 6)
# plt.ylim(0, 20)
# plt.plot(x, y3, 'r-')
# plt.subplot(224)
# plt.xlim(0, 6)
# plt.ylim(0, 20)
# plt.plot(x, y4, 'r-')
# plt.show()

# -------------绘制导数（切线）-------------
def f(x):  # 定义了一个返回值函数
    return c * (x - a) ** 2 + b


# 函数中参与运算的数据类型是什么类型,结果就是什么类型
y = f(x)  # 得到结果是一个矩阵
x1 = 1
y1 = f(x1)  # 得到结果是一个值
x2 = 2
y2 = f(x2)
# 通过 x1,y1,x2,y2 求取w(斜率)  b(截距)
# y = w * x + b
w = (y1 - y2) / (x1 - x2)
b = y1 - w * x1
# 切线方向上y的值
y_slope = w * x + b
plt.plot(x, y, 'r-')
# 绘制切线
plt.plot(x, y_slope, 'b--')
plt.show()
