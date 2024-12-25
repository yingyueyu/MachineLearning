import numpy as np

a = np.arange(15)
print(a)
# 数组的形状， 各个维度的数量
print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype)
print(a.itemsize)

# 改变形状
b = a.reshape(3, 5)
print(b.shape)
print(b)
print(b.ndim)
print(b.size)
print(b.dtype)
print(b.itemsize)

a = np.arange(5, dtype=np.int64)
a = np.arange(5, dtype='i8')
print(a.dtype)
a = np.arange(5, dtype=np.float64)
a = np.arange(5, dtype='f8')
print(a.dtype)
a = np.array(['a', 'b', 'c'], dtype='S1')
print(a.dtype)
a = np.array([True, False], dtype=np.bool)
print(a.dtype)