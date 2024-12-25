# 方法一：转二进制，逐位统计为1的数量
number1 = int(input("请输入一个正整数："))
bin = bin(number1)
x = 0
for o in bin:  # 二进制的数可以直接用for循环迭代获取每一位数字
    if o == '1':
        x += 1
print(bin, x)


# 方法二： 用1与操作来统计1的数量， 比较后通过右移排除掉这位
n = int(input("请输入一个整数："))
count = 0
while n: # 数字为0时循环结束
    count += n & 1 # 末尾如果是1，累加1
    n >>= 1  # 最右边的这位数字已经统计过，右移抛弃掉
print(count)



