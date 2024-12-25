# pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

# x坐标数组
xp = np.array([1, 10])
# y坐标数组
yp = np.array([2, 6])
# 设置图形的数据
plt.plot(xp, yp)  # 默认样式， 蓝色实线
# plt.plot(xp, yp, 'r--')  #  红色虚线
# plt.plot(xp, yp, color='red', linewidth=5, linestyle='--')  # 红色虚线，线宽5
# 显示图形
plt.show()


