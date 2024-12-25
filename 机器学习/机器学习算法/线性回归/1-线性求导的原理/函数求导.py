import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 6, 20)

a = 3
b = 2
c = 2


def f(x):
    return c * (x - a) ** 2 + b


# 手动求导数
def slope(x):
    return 4 * (x - 3)


y = f(x)
x1 = 3
w = slope(x1)
y1 = f(x1)
# y = w * x + b => b = y - w * b
b = y1 - w * x1
y_slope = w * x + b
plt.plot(x, y, 'r-')
plt.plot(x, y_slope, 'b--')
plt.show()
