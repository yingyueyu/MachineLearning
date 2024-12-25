sums = 0  # 累加每一个数的和
for i in range(1, 11):
    count = 1  # 每一个数阶乘的结果
    for j in range(1, i + 1):
        count *= j
    sums += count
print("1到10的阶乘之和:", sums)


