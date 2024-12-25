# 安裝numpy库： pip install numpy
import numpy as np

# 参数是一维的List
arr1 = np.array([1, 2, 3])
print(arr1)
arr1 = np.array(range(5))
print(arr1)


# 参数是二维list, 创建出二维数组
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr2)

# ndmin 指定生成的数组的维度， 第一个参数只作为多维数组的一个元素
arr3 = np.array([1, 2, 3, 4, 5], ndmin=2)
print(arr3)  #  [[1 2 3 4 5]]


# 创建3行2列的空数组， 它dtype设置的类型随机生成数据
# dtype的设置： np.int32; np.int64; np.float32;  np.float64; 'i4'(4字节整数); 'f4'（4字节小数）;  'S1'(1位字符串);  'S2'
arr4 = np.empty((3, 2), dtype=np.int32)
print(arr4)

# 生成全0的数组， 如果生成一维数组，shape参数不需要元组，直接传数组。
# 0也是有整数和小数的表示方式，所以zeros方法也可以数据类型
# arr5 = np.zeros(5, dtype='i4')
arr5 = np.zeros((5,), dtype='i4')
print(arr5)
arr6 = np.zeros((3, 3, 3), dtype='f4')
print(arr6)

# 创建全为1的一维数组， 默认数据类型是浮点数
arr6 = np.ones(5)
print(arr6)
arr7 = np.ones((2, 3), dtype=np.int32)
print(arr7)

# 创建一个象某个数组一样（形状，数据类型）的全零的数组
arr7 = np.zeros_like(arr2)
print(arr7)
arr8 = np.ones_like(arr2)
print(arr8)

# 创建跟某个数组一样的数组，形状和数据都一样
arr9 = np.asarray(arr6)
print(arr9)

# 通过迭代器生成数组， 必须指定数据类型dtype， 否则要报异常
it = iter([1, 2, 3, 4, 5])
arr10 = np.fromiter(it, dtype=np.float32)
print(arr10)

# 生成等差数列
# 从start 到 stop参数的区间，平均分为num份， 返回每一个点的坐标
arr11 = np.linspace(-1, 1, 10)
print(arr11)

# 等比数列
# 生成以base为底，  start到stop之间取num个数作为指数
# 本例的结果： 10^1   10^1.25    10^1.5   10^1.75  10^2
arr12 = np.logspace(start=1, stop=2, num=5, base=10)
print(arr12)

# 返回正态分布数据
arr13 = np.random.randn(3, 2)
print(arr13)

# 产生0~1之间的随机数数组
arr14 = np.random.rand(3, 2)
print(arr14)

#arange()生成指定区间的从0开始的顺序数组
arr15 = np.arange(12)
print(arr15)