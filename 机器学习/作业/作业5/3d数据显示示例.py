import numpy as np
from matplotlib import pyplot as plt

X = np.random.rand(10, 3)
labels = np.random.randint(0, 2, (10, 1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=100, c=labels, cmap='bwr')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.colorbar(sc)
plt.show()
