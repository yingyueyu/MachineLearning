import numpy as np
import matplotlib.pyplot as plt

# 设置均值和标准差
mean = 0.5
std_dev = 0.2

# 生成数据点
x = np.linspace(0, 1, 1000)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# 绘制高斯分布曲线
plt.plot(x, y, label=f'Gaussian Distribution\nMean={mean}, Std Dev={std_dev}')
# plt.xlim(-10, 10)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Gaussian Distribution')
plt.legend()
plt.grid(True)
plt.show()
