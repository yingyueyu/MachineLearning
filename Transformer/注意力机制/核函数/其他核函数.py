import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 1000)


# 高斯核函数
def gaussian(x, sigma=0.5):
    return np.exp(-1 * np.sqrt(x ** 2) ** 2 / 2 * sigma ** 2)


# 矩形核函数
def boxcar(x):
    return np.where(np.sqrt(x ** 2) <= 1, 1, 0)


# Epanechnikov核函数
def epanechnikov(x):
    return np.maximum(0, 1 - np.sqrt(x ** 2))


fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].plot(x, gaussian(x, sigma=3))
ax[1].plot(x, boxcar(x))
ax[2].plot(x, epanechnikov(x))

plt.show()
