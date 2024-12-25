import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)


def get_y(mu, sigma_pow2, x):
    a = 1 / (sigma_pow2 * 2 * np.pi)
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma_pow2))


def get_xy(sigma_pow2, x, y):
    a = 1 / (sigma_pow2 * 2 * np.pi) ** 0.5
    return a * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_pow2))


# y1 = get_y(0, 0.2, x)
# y2 = get_y(0, 1.0, x)
# y3 = get_y(0, 5.0, x)
# y4 = get_y(-2, 0.5, x)

print(get_xy(0.7, np.array([[-1, 0, 1]]), np.array([0, 0, 0])))

# plt.xlim(-5, 5)
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.plot(x, y4)

# plt.show()
