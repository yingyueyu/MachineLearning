l1 =[23, 'abc', 2.17, True]

# 用正数作为下标，从左到右依次是：0， 1， 2.。。
print(l1[0])
# 用负数作为下标，从右到左依次是：-1， -2， -3 。。。
print(l1[-1], l1[-3])

# 遍历
for temp in l1:
    print(temp)


# 切片方式访问
print(l1[::])  # 不设任何参数，返回原列表
print(l1[::-1])  # -1步长逆序
print(l1[:2])
print(l1[1:])
print(l1[1:3])

#================增加==================

# 在列表的末尾添加元素
l1.append('小红')
print(l1)

l2 = [6, 8]
# l1.append(l2)
print(l1)

# 把参数列表里的元素依次添加到原列表
l1.extend(l2)
print(l1)

# 在指定下标位置（第一个参数）插入元素（第二个参数）
# l1.insert(1, '二哥')
# insert的下标也可以用负数
l1.insert(-6, '二当家')
print(l1)

#====================删除====================
print(l1)  # [23, '二当家', 'abc', 2.17, True, '小红', 6, 8]
# 移除最后一个元素并返回
print(l1.pop())  #  8
print(l1)  # [23, '二当家', 'abc', 2.17, True, '小红', 6]
# l1.pop(4)
# 下标可以用负数
l1.pop(-3)
print(l1)

# 用del语句删除列表指定元素
del l1[5]
print(l1)

# 删除列表的元素，参数用下标或具体的值都可以
l1.remove(2.17)
print(l1)

# ==================修改==================
l1[1] = '大当家'
print(l1)
# 通过修改遍历出来的变量，是不会改变原来列表的
for tmp in l1:
    tmp = 'xxx'
print(l1)

# ===============查找======================
if 'abc' in l1:
    print('存在')
else:
    print('不存在')

if 25 not in l1:
    print('不存在')
else:
    print('存在')


# 其它方法
print(l1.index('abc'))  # 返回指定元素的下标，也可以设start和end参数
# l1.clear()  # 清空列表
l2 = l1.copy()  # 复制新的列表
l1.reverse()  # 颠倒元素的顺序
print(l1)

l1 = [5, 6, 8, 2, 54, 1, 98]
l1.sort()
print(l1)
l1.sort(reverse=True)
print(l1)

#  练习：
#  两个列表：  l1 = [1, 5, 7, 9, 12]    l2 = [5, 9, 12, 23, 55]
#  找出l1中存在，但是l2中不存在的元素并打印
# l1 = [1, 5, 7, 9, 12]
# l2 = [5, 9, 12, 23, 55]
# # for lis2 in l2:
# #     if lis2 in l1:
# #         l1.remove(lis2)
# # print(l1)
#
# for i in l1:
#     if i not in l2:
#         print(i)

