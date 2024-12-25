import matplotlib.pyplot as plt
import numpy as np
#
# # 抛物线
# x1 = np.linspace(-10, 10, 30)
# # print(x1)
# y1 = x1 ** 2
# # 生成子坐标系， 行列和序号的参数。
# # 同一张图里面的坐标系用序号来区分，序号从1开始
# ax1 = plt.subplot(2, 2, 1)
# # 创建子坐标系之后，紧跟的用plt对象画图的操作就是当前坐标系
# # plt.plot(x1, y1)
# # 创建子坐标系时返回坐标系对象，然后用这个对象画图，只针对这个坐标系
# ax1.plot(x1, y1)
#
# # 直线
# x2 = np.linspace(1, 10, 20)
# y2 = 2 * x2
# plt.subplot(2, 2, 2)
# plt.plot(x2, y2)
#
# # 正弦曲线
# x3 = np.linspace(0, 10, 100)
# y3 = np.sin(x3)
# plt.subplot(2, 2, 3)
# plt.plot(x3, y3)
#
# # 余弦曲线
# x4 = np.linspace(0, 10, 100)
# y4 = np.cos(x4)
# plt.subplot(2, 2, 4)
# plt.plot(x4, y4)
#
#
#
# plt.show()

# 一次创建多个坐标系，并返回。 ax包含所有坐标系
fig, ax = plt.subplots(2, 2)

x1 = np.linspace(-10, 10, 30)
y1 = x1 ** 2
ax[0, 0].plot(x1, y1)
x2 = np.linspace(1, 10, 20)
y2 = 2 * x2
ax[0, 1].plot(x2, y2)
x3 = np.linspace(0, 10, 100)
y3 = np.sin(x3)
ax[1, 0].plot(x3, y3)
x4 = np.linspace(0, 10, 100)
y4 = np.cos(x4)
ax[1, 1].plot(x4, y4)
plt.show()