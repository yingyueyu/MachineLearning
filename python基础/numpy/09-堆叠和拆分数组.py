import numpy as np

a = np.floor(10 * np.random.random((2, 2)))
print(a)

b = np.floor(10 * np.random.random((2, 2)))
print(b)

# d = np.floor(10 * np.random.random((3, 2)))
# print(d)

# 垂直方向堆叠， 把两个(多个)数组的行拼接成一个新的数组
# 要求堆叠方向上的形状要一致
# c = np.vstack((a, b))
# print(c)
# print(c.shape)

# 水平堆叠， 把后面数组的行依次拼接到第一个数组每行的后面
# 要求多个数组的行的形状要一致
c = np.hstack((a, b))
print(c)

a = np.floor(10 * np.random.random((4, 6)))
print(a)
# 水平拆分，把原数组拆分为n个数组， 平均分配列。 原数组的列数必须要能整除拆分的数组的个数
b = np.hsplit(a, 2)
# print(b)
# 第二个参数如果是元组， 就以元组设定列的序号作为一个数组，另外的前后两部分各为一个数组，共返回三个数组
b = np.hsplit(a, (2, 5))
print(b)

a = np.floor(10 * np.random.random((6, 3)))
print(a)
b = np.vsplit(a, 3)
# print(b)
b = np.vsplit(a, (2, 5))
print(b)


