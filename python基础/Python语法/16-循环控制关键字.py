
# 打印0~99， 循环变量为11时，终止循环，后面的循环不再执行，
# 11这一次循环break之后的print也不执行。所以只能打印到10
for i in range(100):
    if i == 11:
        break
    print(i)


# 从键盘输入数字， 从列表中查找这个数字，找到了就打印序号并停止查找；如果找不到就打印没找到
#
#
# flag = False # 找到的标志
# i = 0  # 循环计数器
# n = input("输入数字：")
# lis = [21, 45, 55, 88, 102]
# for num in lis:
#     i += 1
#     if int(n) == num:
#         flag = True  # 找到了设置循环标志
#         break # 找到后就终止循环
# # 循环结束后再来打印结果
# if flag:
#     print(f"恭喜找到！！！, 是第{i}个数字")
# else:
#     print("输入不存在")



# 打印1~100之间的偶数
for i in range(1, 100):
    if i % 2 != 0: # 奇数，用continue跳过当次循环，继续下一次循环
        continue
    print(i)


i = 10
if i >= 10:
    print("********")
else:
    pass


