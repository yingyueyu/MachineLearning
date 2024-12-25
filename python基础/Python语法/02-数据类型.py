# 整数int
i = 0
print(i)
j = 100
print(j)
k = 123456789123456789000000000000000011
print(k)
m = -88
print(m)

# 小数 float
f1 = 3.1415
print(f1)
f2 = -0.009
print(f2)
f3 = 5.987654321654987
print(f3)
print(0.123456 + 0.000612)

# 复数
#  定义方式一： 实部 + 虚部j
c1 = 20 + 10j
c2 = 20 + 10J
print(c1, c1.real, c1.imag)
print(c2, c2.real, c2.imag)

# 方式二： 用函数complex(实部, 虚部)
c3 = complex(5, 10)
print(c3, c3.real, c3.imag)

# 虚部参数可省略，默认就是0
c4 = complex(10)
print(c4, c4.real, c4.imag)

# 字符串
s1 = '字符串'
s2 = "this is string"
print(s1, s2)
s3 = '=*' * 10 # 重复字符串多少次
print(s3)

# List
'''
    注意：
    多个元素用逗号隔开
    可用下标来读写列表的元素
    两个list可以用+做拼接
    list里的元素的类型可以不一致
'''
l1 = [1, 3, 5]
l2 = [2, 4, 6]
print(l1,  l1[0])
print(l1 + l2)
l3 = [1, 3.14, True, 'abc',[4, 6, 8]]
print(l3)

# tuple
# tuple的元素是不能改变，list的元素可以改变
t1 = (1, 3, 5, 'abc', 3.14)
print(t1)
print(t1[3])
# t1[3] = 'xyz' # 不能对元组的元素赋值，会报异常
# 定义一个元素的元组，必须要在元素后面加个逗号，否则它会把小括号当成四则运算的括号
t2 = (8,)
print(t2)
print(t2[0])
# 定义没有元素的元组
t3 = ()
print(t3)

# dictionary
dict1 = {}
print(dict1)

dict2 = {"李明": 95, "张山": 90}
print(dict2)
print(dict2["李明"])
# 字典的key要用能唯一确定的数据类型，比如数字，字符串，元组。 但是list不行
# dict3 = {[1, 3, 5]: 20, [2, 4, 6]: 10}
# print(dict3)

# set
# 无序， 不重复
set1 = {1, 3, 5}
print(set1)
# 空的集合不能用{}， 因为{}用来定义空的字典的
set2 = set()

print(type(i))  # int
print(type(f1)) # float
print(type(s1)) # str
print(type(True)) # bool
print(type(l1)) # list
print(type(t1)) # tuple
print(type(dict1)) # dict
print(type(set1)) # set