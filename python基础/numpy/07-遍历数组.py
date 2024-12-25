import numpy as np

l = [0, 1, 2, 3, 4]
print(l)

a = np.arange(5)
print(a)
# 一维数组遍历出来就是元素
for i in a:
    print(i)


b = np.arange(20).reshape(4, -1)
print(b)
# 多维数组的遍历， 取出第一个维度的所有元素，返回数组（原来的维度-1）
for x in b:
    print(x)

# b.flat 把多维数组展开为一维数组。
for item in b.flat:
    print(item)

# 展开三维数组
c = np.arange(48).reshape(4,3,-1)
print(c)
for y in c.flat:
    print(y)