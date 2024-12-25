import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)
# 把原数组的列依次变为新数组的行，这个操作称为转置
print(a.transpose())
# .T也是做转置运算
print(a.T)


# 两个数组， 求每一位乘积的和
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
# 把其中一个转置， 利用矩阵的点积运算dot()
print(np.dot(a.T, b))