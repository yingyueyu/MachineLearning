tup1 = ('Google', 'HQYJ', 666, 3.14)
# 元组只有一个元素，后面需要一个逗号（避免当成四则运算）
tup2 = (888, )
tup3 = ()

print(tup1[1])
print(tup2[0])
print(type(tup3))
# 也可以切片访问
print(tup1[::-1])
for tmp in tup1:
    print(tmp)


# =============修改========
tup4 = (11, 22, 33)
tup5 = ('abc', 'xyz')
# 重新给元组的变量赋值是可以
tup4 += tup5
print(tup4)
# 不能对元素重新赋值
# tup4[0] = 99

tup6 = (1, 'abc', [1, 2, 3])
# 元素对象里面的内容是可以改
tup6[2][0] = 999
print(tup6[2])

#===============删除=========
del tup6 # 删除变量
# print(tup6)

#============运算符或方法
a = (1, 2, 3)
b = (4, 5, 6)

print(len(a))
c = a + b
print(c)
a += b
print(a)
tup = ('Hi!', 'Hello') * 4
print(tup)

if 3 in (1, 2, 3):
    print('True')

for x in (1, 2, 3):
    print(x, end='')
