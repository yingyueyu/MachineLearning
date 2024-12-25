import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(20)
y = np.random.random(20)

# s   点的大小
# c   颜色。  如： 'r',  'red',   '#FF0000'
plt.scatter(x, y, s=40, c='r', marker='1')
plt.show()