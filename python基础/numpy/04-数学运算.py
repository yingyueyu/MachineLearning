import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 加法运算是对应元素相加
print(a + b)
print(a - b)

# 乘法运算对应位乘法
print(a * b)
print(a / b)

# 平方根
print(np.sqrt(a))
# e的多少次方
print(np.exp(a))

# a = np.arange(12).reshape(3, 4)
a = np.arange(1, 7).reshape(2, -1)
b = a
print(a)
print(b)
print('a+b', a + b)
print('a-b', a - b)
print('a*b', a * b)
print('a/b', a / b)
print(np.sqrt(a))
print(np.exp(a))

print(a)
# 求和， 所有元素相加
print(a.sum())
# axis=0  纵向（列）汇总
print('axis=0', a.sum(axis=0))
# axis = 1 横向（行）汇总
print('axis=1', a.sum(axis=1))
print(a.max())
print(a.min())
print(a.mean())
