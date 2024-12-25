# 从1数到5
i = 1
while i < 6:
    print(i)
    i += 1

# 求1加到100的和
i = 0
sums = 0
while i < 100:
    i += 1
    # print(i, end=' ')
    sums = sums + i
print(sums)

# 循环嵌套
i = 0
while i < 3:
    j = 0
    while j < 2:
        print(f'{i}, {j}')
        j += 1
    i += 1

# 用while循环打印三角形的星号
i = 1
while i < 6:
    j = 0
    while j < i:
        print("*", end="")
        j += 1
    print("")
    i += 1





# j = 0
# while j < 6:
#     j += 1
#     print("*" * j)

