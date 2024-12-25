import numpy as np
import math


def gauss_kernel(size, sigma):
    if size % 2 != 1:
        return
    half = size // 2
    x = np.arange(-half, half + 1).astype(np.int32)
    y = x.copy()
    x, y = np.meshgrid(x, y)
    a = 1 / (2 * np.pi * sigma ** 2)
    return a * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


kernel = gauss_kernel(3, 0.8)
print(kernel)
