import matplotlib.pyplot as plt
import numpy as np

x = np.array(['zhangsan', 'lisi', 'wangwu', 'Jack'])
y = np.array([88, 66, 99, 100])

# width/height 柱子的宽度，0~1的小数
# color 柱子的颜色， 可以统一设置颜色，也可以用数组分别设置颜色
# plt.bar(x, y, width=0.5, color=['red', 'yellow', 'blue', 'green'])  # 竖向的柱状图
plt.barh(x, y, height=0.5, color=['red', 'yellow', 'blue', 'green'])  # 横向柱状图
plt.show()