# 方法一：循环,判断偶数
j = 0
for i in range(101):
    if (i % 2) != 0:
        continue
    j += i
print('1到100（包含100）的偶数之和为：', j)

# 方法二： for循环步长为2
sums = 0
for i in range(0, 101, 2):
    sums += i
print(sums)