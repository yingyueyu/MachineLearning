# 计算两数之和并返回
def add(a, b):
    total = a + b
    return total


print(add(3, 2))


def mul(a, b):
    total = a * b
    return total


print(mul(52, 10))


# 阶乘。 递归调用
def fact(n):
    if n == 1:
        return n
    else:
        return n * fact(n - 1)


print(fact(3))

