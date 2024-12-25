import numpy as np

# 没有复制数组（使用同一个数组）
a = np.arange(12)
b = a
print(b is a)
# 对形状属性赋值可以直接多个数字 3, 4  或元组都可以(3, 4)
b.shape = (3, 4)  # 3, 4
print(b)
print(a.shape)


# 浅拷贝.  形状相互不影响，但是数据会影响
a = np.arange(12)
c = a.view()
print(c is a)
print(c.base is a)
c.shape = (2, 6)
print(c)
print(a.shape)
c[0][0] = 99
print(a)
# 切片的数组也是浅拷贝
s = a[3:5]
print(s)
# 切片方式对数组赋值
s[:] = 88
print(a)

#深度拷贝
d = a.copy()
print(d is a)
print(d.base is a)
del a
d[4:6] = 100
print(d)
