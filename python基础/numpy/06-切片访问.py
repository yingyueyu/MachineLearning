import numpy as np

a = np.arange(10)
print(a)
print(a[2: 7: 2])
# 切片的下标也可以用slice函数创建
# b = slice(2, 7, 2)
# print(a[b])

print(a[::])
print(a[:])
print(a[1::])
print(a[:-5:-1])
print(a[::-1])


a = np.arange(16).reshape(4, 4)
print(a)
print(a[::])
print(a[:])
# 多维数组切片就要多个切片参数，参数之间用逗号分开
# 0~2行， 0~2列
print(a[0:2, 0:2])
# 0~2行，所有列
print(a[0:2, :])

# 1~3行， 1~3列
print(a[1:3, 1:3])
# 0~3行， 从0列开始，步长为2
print(a[0:3, 0::2])

# 省略号(...)的用法: 所有行（或列）， 等价于:(或::)
# 截取第二列
print(a[..., 1])
print(a[1, ...])
print(a[..., 1:])



a = np.arange(64).reshape(4, 4, 4)
print(a)
# print(a[1, 1, 1])
print(a[..., 1:3 , 1:3])