str1 = 'hello hqyj aaaaa'

# 从0开始找
i = str1.find('h')
# start参数， 从指定位置开始查找，一直找到字符串结束
i = str1.find('h', 3)
# end 结束位置， 从下标3到8区间查找， 左闭右开（不包含最后位置的字符）
i = str1.find('h', 3, 6)
print(i)

# 如果查找不到，会抛异常
# i = str1.index('hx')
# print(i)

i = str1.rfind('h')
print(i)

i = str1.rindex('h')
print(i)

# 字符串转小写字母
print(str1.lower())
# 字符串转大写
print(str1.upper())
# 对每个单词的首字母转大写
print(str1.title())

# 以指定字符串开头
print(str1.startswith('hello'))
# 以指定字符串结束
print(str1.endswith('aaaaa'))
# 所有字符是否是空格
print(' '.isspace())
# 所有字符是否是数字
print('8848'.isdigit())
# 所有字符是否是字母，不区分大小写
print('abcCDE'.isalpha())
# 所有的字符是否是字母或数字
print('66abc123cc'.isalnum())


# 键盘输入一个字符串， 判断字符串里面有多少个空格，多少个字母，多少个数字，多少个标点符号
# str1 = input('输入字符串：')
# space = 0 # 空格的数量
# alpha = 0 # 字母的数量
# num = 0  # 数字的数量
# x = 0  # 其它符号的数量
# for i in str1:
#     if i.isspace():
#         space += 1
#     elif i.isalpha():
#         alpha += 1
#     elif i.isdigit():
#         num += 1
#     else:
#         x += 1
# # x = len(str1) - alpha - num - space
# print(f'空格：{space}，字母：{alpha}，数字：{num}，标点：{x}')


str2 = 'hello world hello china'
# 以指定的字符串把原字符串分割前，自己后后面部分共三个字符串，以元组的方式返回。
# 如果没有找到指定字符串，返回的元组里面的第一个字符串就是原字符串，后面两个是空字符串
pt = str2.partition('hello')
print(pt)
print(type(pt))

# 功能类似partition, 查找顺序是从右到左。 返回的三个字符串的顺序还是按原字符串中的顺序
rpt = str2.rpartition('hello')
print(rpt)

# 把原字符串按指定字符串分割成多个字符串，以列表对象返回
# 分割字符串必须有内容（不能以空字符串作为分隔符）； 分割字符串不会出现在结果中
sp = str2.split(' ')
print(sp)
print(type(sp))


str3 = '把原字符串\n按指定字符串\n分割成多个字符串'
# 按照行读取字符串，返回字符串列表
spl = str3.splitlines()
print(spl)


# 有一个字符串： http://hqyj.com?userId=13546985
# 从url分别获取参数名和参数值
i = 'http://hqyj.com?userId=13546985'
j = i.index('?')
para = i[j+1:]
l = para.split('=')
print(f'参数名： {l[0]}, 参数值： {l[1]}')


# 'hello world hello china'
str2 = 'hello world hello china'
print(str2.count('o'))
print(str2.count('o', 4, 15))

# 替换指定的字符串, 默认替换所有找到的字符串
print(str2.replace('hello', 'HELLO'))
# count参数设置替换的次数
print(str2.replace('hello', 'HELLO', 1))

# 句子（整个字符串）的首字母大写
print(str2.capitalize())

l2 = ['abc', 'def', 'xyz']
joinStr = '_'
# 用一个指定的字符串，对列表（可迭代的多个字符串）进行连接，返回一个字符串，如： abc_def_xyz
print(joinStr.join(l2))










# 键盘上输入一个文件名， 把这个文件名改为一个随机的名字（数字），扩展名不能变