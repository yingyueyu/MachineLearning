import numpy as np

# np.rollaxis(a, axis, start)
#  a   数组
#  axis  要滚动的轴的序号
#  start  滚动到的位置
# a = np.arange(24).reshape(2, 3, 4)
# print(a)
# where()返回元素的坐标， 参数条件表达式
# print(np.where(a == 6)) # [1, 1, 0]
# print(np.where(a == 2)) # [0, 1, 0]

# b = np.rollaxis(a, 2, 0) # start参数省略，默认为0， 滚动到最前面
# # print(np.where(b == 6)) # [0, 1, 1]
# # print(np.where(b == 2)) # [0, 0, 1]
# # print(b)
# print(b.shape)
#
# c = np.rollaxis(a, 2, 1)
# print(c.shape)
#
# d = np.rollaxis(a, 1, 2)
# print(d.shape)
#
# print(np.rollaxis(a, 1, 0).shape)
#
# print(np.rollaxis(a, 0, 1).shape)
#
# print(np.rollaxis(a, 0, 2).shape)
# print(np.rollaxis(a, 0, 3).shape)


a = np.arange(8).reshape(2, 2, 2)
# a = np.arange(24).reshape(2, 3, 4)
print(a)
print(a.shape)
print(np.where(a==0))
print(np.where(a==1))
print(np.where(a==2))
print(np.where(a==3))
b = np.swapaxes(a, 0, 2)
print(b)
print(b.shape)
print(np.where(b==0))
print(np.where(b==1))
print(np.where(b==2))
print(np.where(b==3))


