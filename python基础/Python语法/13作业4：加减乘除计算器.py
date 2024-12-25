# a = int(input("请输入第一个数："))
# c = input("请输入运算符:")
# b = int(input("请输入第二个数："))
# if c == '+':
#     print(a + b)
# if c == '-':
#     print(a - b)
# if c == '*':
#     print(a * b)
# if c == '/':
#     print(a / b)


a = input("请输入第一个数：")
c = input("请输入运算符:")
b = input("请输入第二个数(除数不能0)：")
a = eval(a + c + b)
print(a)

