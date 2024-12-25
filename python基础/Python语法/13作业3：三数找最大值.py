a = int(input("请输入一个数："))
b = int(input("请输入一个数："))
c = int(input("请输入一个数："))
# if a >= b and a >= c:
#     print(a)
# elif b >= a and b >= c:
#     print(b)
# else:
#     print(c)

# max = a
# if a > b:
#     if a > c:
#         max = a
#     elif a < c:
#         max = c
#     else:
#         max = a
# elif a < b:
#     if b > c:
#         max = b
#     elif b < c:
#         max = c
#     else:
#         max = b
# else:  # a == b
#     if c > b:
#         max = c
#     elif c < b:
#         max = b
#     else:
#         max = c
# print(max)


# temp = a
if b > a:
    a = b
if c > a:
    a = c
print(a)


