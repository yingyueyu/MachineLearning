import numpy as np

a = np.array([[1, 2], [3, 4]])
a = np.concatenate((a[:, 0, None], a, a[:, 1, None]), axis=1)
a = np.concatenate((a[0, None, :], a, a[1, None, :]), axis=0)
print(a)
