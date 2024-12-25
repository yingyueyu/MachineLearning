import numpy as np

a = (np.arange(36) ** 2).reshape(6, 6)
print(a)
print(a.sum())
print(a.mean())
print(a.max())
print(a.min())
# 按列统计
print(a.sum(axis=0))
print(a.mean(axis=0))
print(a.max(axis=0))
print(a.min(axis=0))
# 按行统计
print(a.sum(axis=1))
print(a.mean(axis=1))
print(a.max(axis=1))
print(a.min(axis=1))


print(a[2:5, 1:5])


b = np.hsplit(a, (2, 3))
print(b)

