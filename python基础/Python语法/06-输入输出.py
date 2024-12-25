# 直接打印字符串或字符串表达式
print("Hello Python")
s = "Python"
print("Hello: " + s)

# 打印多个参数
a = 3
b = 2
print('a + b =', a + b)

# 用%格式化输出:   %s  字符串； %d   整数；   %f   小数
print("%d + %d = %d" % (a, b, a + b))

# f格式化输出
print(f'{a} + {b} = {a + b}')

# end参数: 默认是\n,所以print每打印一次会换行。 这个end参数可以设置为想要的任意值
print("end参数", end=" ")

n = input("请输入一个数字：")
print(type(n))
n = int(n) # 强制类型转换为整型
print(type(n))