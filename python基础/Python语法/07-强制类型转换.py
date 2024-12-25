# int()
a = "100"
# 第二个参数是指定要转换的这个数字原来是什么进制，默认是十进制。 其它的进制还有2, 8, 16
b = int(a, 10)
print(b)

# str()
i = 88
print("数字转字符串：" + str(i))

# float()
pi = "3.14"
r = 5
print(float(pi) * r ** 2)

# list()  <=>  set()
# list转set, 再转回list， 实现去重的效果
l = [1, 3, 5, 3, 7]
set1 = set(l)
l1 = list(set1)
print(l1)

# eval()
s1 = "3 + 2"  # eval()计算表达式
a = eval(s1)
print(a,  type(a))
s2 = "print(\"abc\")"  #  eval()执行print()方法
eval(s2)
s3 = "[1, 3, 5]"  # eval() 执行list定义语句，返回list对象
l = eval(s3)
print(l, type(l))

# hex() 把十进制转为十六进制， 十六进制数以0x开头；  八进制0o开头； 二进制0b开头
print(hex(256))
print(oct(64))
print(bin(15))

# 把字母转ascii码值
print(ord('A'))