import matplotlib.pyplot as plt
import numpy as np

# 给出的坐标点如果不在一条直线上，那么绘制的就是折线
x = np.array([0, 1, 2, 3, 4])
y = np.array([10, 6, 7, 3, 5])
plt.plot(x, y)
# x可以省略， 如果不设置x, 那么自动使用从0开始的整数序列做为x的点
# plt.plot(x)
plt.show()


