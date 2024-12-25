# 定义一个列表，随机产生10个10以内的整数，添加到这个列表中， 把列表中重复的数字去掉
import random

# set1 = set(random.randint(1, 10) for i in range(10))
# list1 = list(set1)
# print(list1)


list1 = []
for i in range(10):
    num = random.randint(1, 10)
    list1.append(num)
# print(list1)
set1 = set(list1)  # list 转 set , 目的是去重
list1 = list(set1)  #  set 转 list
print(list1)