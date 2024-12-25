import numpy as np

a = np.arange(12)
print(a)
b = a.reshape(3, 4)
print(b)
b = a.reshape(6, 2)
print(b)
b = a.reshape(2, -1)
print(b)
# resize()直接改变原数组的形状，功能跟reshape()一样
a.resize(3, 4)
print(a)