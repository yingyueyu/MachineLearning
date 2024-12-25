# def fun(index, lst):
#     # try:
#         # if lst[index] in lst:
#     print(f'找到了!结果为{lst[index]}')
#     # except IndexError:
#     #     print('抱歉,没有找到！')
#
#
# import random
# lst1 = [random.randint(1, 50) for _ in range(20)]
# print(lst1)
# while True:
#     try:
#         data = int(input('请输入列表下标:'))
#         fun(data, lst1)
#         break
#     except ValueError:
#         print('请你输入整数!')
#     except IndexError:
#         print('请输入-20 ~ 19之间的整数')
#     except BaseException:
#         print('发生其它错误，请重试')


def multiplication(a, b):
    print(f'{b}*{a}={a * b}', end='\t')
    if a == b:
        print()

# 方式一： lambda
list1 = [(lambda a, b: print(f'{b}*{a}={a * b}'))(i, x) for i in range(1, 10) for x in range(1, i + 1)]

# 方式二：函數
print('函数')
list2 = [multiplication(i, x) for i in range(1, 10) for x in range(1, i + 1)]