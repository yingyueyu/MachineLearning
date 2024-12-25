import matplotlib.pyplot as plt
import numpy as np

# 产生1000个正太分布的点
x = np.random.randn(1000)

# bins   设置柱子的数量； alpha透明度，取值范围0~1
plt.hist(x, bins=40, alpha=0.5)

plt.show()