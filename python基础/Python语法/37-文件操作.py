
# 打开文件
# f1 = open('data.txt', 'r', encoding='utf-8')
# txt = f1.read() # 读取内容
# print(txt)
# f1.close()  # 关闭文件


# 用with关键字打开文件，语句块结束后文件会自动关闭
# with open('data.txt', 'r', encoding='utf-8') as f2:
#     txt = f2.read()
#     print(txt)
#

# readline每次读一行，按迭代器方式读取
# with open('data.txt', 'r', encoding='utf-8') as f3:
#     line = f3.readline()
#     print(line)
#     line = f3.readline()
#     print(line)
#     line = f3.readline()
#     print(line)
#     line = f3.readline()
#     print(line)


# 一次读取所有行的内容，按行转换为list返回（每个元素是一行）
# with open('data.txt', 'r', encoding='utf-8') as f4:
#     lines = f4.readlines()
#     print(lines)


with open('data.txt', 'a+', encoding='utf-8') as f5:
    f5.write('写入新的内容\n 第二行的内容\n')
    # seek方法移动文件读取的指针。 因为写操作结束后文件访问指针已经到了文件末尾，所读不了内容
    # 只有把指针移到开头，才能再次读取
    f5.seek(0)
    print(f5.read())



