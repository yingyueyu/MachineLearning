# def add(a, b):
#     return a + b


# lambda表达式赋给一个变量，这个变量相当于是一个函数
add = lambda a, b: a + b

print(type(add))
print(add(10, 20))
# (lambda a, b: a + b) 是一个匿名函数
# (20, 30)  调用函数的参数列表
print((lambda a, b: a + b)(20, 30))


# 练习：下面的遍历代码改写为匿名函数方式
list1 = [10, 20, 30, 40, 50]
for i in range(len(list1)):
    print(list1[i])

# 匿名函数根据下标返回元素
getItem = lambda i: list1[i]
for i in range(len(list1)):
    print(getItem(i))


# 字典排序
stus = [
            {'name':'张三', 'score': 90},
            {'name':'李四', 'score': 80},
            {'name':'小红', 'score': 85},
            {'name':'小军', 'score': 70}
        ]
# key参数给了匿名函数， 根据每个字典返回分数，排序就按分数来排
stus.sort(key=lambda d: d.get('score'), reverse=True)
print(stus)

