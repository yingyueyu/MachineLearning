# 算术运算符
a = 10
b = 20
print(a + b)
print(a - b)
print(a * b)
# 除法返回结果是小数，即便可以整除也是小数
print(b / a)
print(type(b / a))
# 整数整除的结果是正数
print(9 // 2)
# 小数整除得到的结果是小数
print(9.0 // 2.0)
print(b % a)
print(a ** b)

# 赋值运算符
a += b
print(a)
a -= b
print(a)
a *= b
print(a)
a /= b
print(a)
a %= b
print(a)
a **= b
print(a)
a //= b
print(a)

# 同时给多个变量赋值
i, j, k, s = 10, 20, 30, '王五'
print(s)
l = [1, 3, 5]
l[0], l[2] = l[2], l[0]  # 一个语句完成列表元素的交换
print(l)


# 比较运算符
a = 10
b = 20
print(a == b)
print(10 == 10)
print(a == 10)
print(a == (2 + 8))
s1 = str("abc")
s2 = 'ab' + 'c'
print(s1 == s2)

print(a != b)

print(a > b)

print(a < b)

print(a >= 10)  #  a > 10   或者   a == 10
print(a <= 10)

# 逻辑运算符
print(a > 10 or a == 10)
print(a > 10 and a == 10)
print(not a > 10)

# 位运算
a = 60
b = 13
print(a & b)  # 12
print(a | b)  # 61
print(a ^ b)  # 49
print(~a)  # -61
print(a << 2)  # 240
print(a >> 2)  # 15
