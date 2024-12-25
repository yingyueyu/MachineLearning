# # 列表推导式
# list1 = []
# for i in range(10):
#     list1.append(i * 2)
# print(list1)
#
# def fn(x):
#     return x * 2
#
# list2 = [fn(i) for i in range(10)]
# list2 = [2*i for i in range(10)]
# print(list2)
#
# # 推导表达式用lambda函数，匿名函数用小括号括起来，后面还有一队小括号，它是调用参数
# list3 = [(lambda x: x * 2)(i)  for i in range(10)]
# print(list3)
#
# # 两个for循环， 第一个相当于外层循环，第二个循环相当于内层循环
# list4 = [str(i) + '_' + s for i in range(5) for s in ['a', 'b', 'c', 'd', 'e']]
# print(list4)
#
#
# animal = ''
# flag = 0
# # if flag == 1:
# #     animal = '猫'
# # else:
# #     animal = '狗'
#
# animal = '猫' if flag == 1 else '狗'
# print(animal)
#
#
# # 有一个数字列表，  小于5显示为0， 大于等于5显示1
# list5 = [1, 2, 3, 4, 5, 6, 7, 5, 8, 3, 2, 0]
# list6 = [0 if i < 5 else 1 for i in list5]
# print(list6)
#
#
#
#
#
# # map方法能遍历第二个参数，用第一个参数表达式处理每一个元素， 最终返回一个map对象， 用list方法把map转为列表
# # res = list(map(lambda x: 2 * x, range(5)))
# res = list(map(fn, range(5)))
# print(res)
#
# # map方法如果有多个列， 不会做多层循环，只遍历一次，每个列表都取对应下标的元素作为运算
# res = list(map(lambda x, y: str(x) + '_' + y  , range(5), ['a','b','c', 'd', 'e']))
# print(res)
#
# # 返回1~100的偶数
# # 第一个参数的函数返回True， 当前元素保留； 如果返回False，当前元素被过滤
# def fn1(x):
#     if x % 2 == 0:
#         return True
#     else:
#         return False
#
# # res = list(filter(fn1, range(1, 101)))
# res = list(filter(lambda x: not x % 2, range(1, 101)))
# print(res)


# 迭代器
list1 = [1, 2, 3, 4, 5]
it = iter(list1)
# print(next(it))
# print(next(it))
# print(next(it))
# print(next(it))
# print(next(it))
# print(next(it))
# for i in it:
#     print(i)


# 创建一个迭代器类， 返回一组数字，[10, 20, 30, 40, 50]
# class MyIter:
#     def __init__(self):
#         self.num = 0  # 初始化一个属性
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         self.num += 10
#         if self.num < 60:
#             return self.num
#         else:
#             raise StopIteration #  迭代完成，抛出终止异常
#
# my_iter = MyIter()
# print(next(my_iter))
# print(next(my_iter))
# print(next(my_iter))
# print(next(my_iter))
# print(next(my_iter))
# # print(next(my_iter))
#
# my_iter2 = MyIter()
# for i in my_iter2:
#     print(i)

#===============yield================
# 传入参数n, 返回从n到0的一个数列

# 函数中用到yield关键字，这个返回的就是生成器（迭代器）对象
# def desc_num(n):
#     while n >= 0:
#         # 返回一个值，返回之后就会停下来（能记住执行位置），直到下一次调用（作用类似迭代器）
#         yield n
#         n -= 1
#
#
# iter = desc_num(5)
#
# print(next(iter))
# print(next(iter))
# print(next(iter))
# print(next(iter))
# print(next(iter))
# print(next(iter))
# # print(next(iter))
#
# iter2 = desc_num(6)
# for i in iter2:
#     print(i)


#   用迭代器打印出斐波那契数列的n以前的所有数字。如： n=5,  打印  1, 1 2 3, 5
# n=1: a=0 , b=1
# n=2: a=1,  b=1
# n=3: a=1,  b=2  (a = b = 1, b = a + b = 1 + 1)
# n=4: a=2,  b=3  (a = b =2,  b = a + b = 1 + 2 = 3)

# def fib(n):
#     a, b = 0, 1
#     for _ in range(n):
#         yield b
#         a, b = b, a + b
#
# iter = fib(12)
# for i in iter:
#     print(i, end=' ')



# 做for in循环同时需要循环变量
# list1 = ['a', 'b', 'c', 'd', 'e']
#
# # for i in list1:
# #     print(i)
# # for i in range(len(list1)):
# #     print(list1[i])
#
# # enumerate封装之后，返回第一个值是从0开始的序号， 第二个值迭代出来的内容
# for index, value in enumerate(list1):
#     print(index, value)
#
# for obj in enumerate(list1):
#     print(obj) # 其实enumerate返回的是序号和元素组成的元组



# list1 = ['a', 'b', 'c']
# list2 = [1, 2, 3, 4]
# list3 = [3.12, 2.15, 8.6, 7.9]
#
# # 拆分列的每一元素， 分别创建元组， 返回一个可迭代对象
# z1 = zip(list1)
# for s in z1:
#     print(s)
#
# # 多个列表压缩， 按长度最短的列表为准，把每一个列表对应下标的元素取出来生成元组
# z2 = zip(list1, list2, list3)
# for s in z2:
#     print(s)
#
# # 练习： 姓名列表：['张三', '李四', '王五'],    成绩列表：      [90, 92, 88]
# # 显示如下：
# # 张三：90
# # 李四：92
# # 王五：88
# l1 = ['吴龙', '顾玉', '吴*鸿', '潘某卫']
# l3 = [80, 90, 85, 80]
#
# z3 = zip(l1, l3)
# for name, score in z3:
#     print(f'{name}: {score}')
#
# # zip压缩的变量前面加星号可以解压
# z4 = zip(l1, l3)
# print(*z4)


# 用zip实现enumerate的功能
# list1 = ['a', 'b', 'c']
# for t in zip(range(len(list1)), list1):
#     print(t)


def fn(*args, **kwargs):
    for arg in args:
        print(arg)
    for kwarg in kwargs.items():
        print(kwarg)


fn('a', 2, 3.14, 'xy', name='Jack', age=20)


