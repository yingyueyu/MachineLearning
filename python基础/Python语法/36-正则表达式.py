# re是正则表达式模块
import re

# # 查找字符串是否存在‘www’， 从第一个字符开始匹配
# res = re.match('www', 'www.hqyj.com')
# print(res)
# print(res.span()) # 返回元组，被查找字符串的起始和节数下标
# print(res.start()) # 开始下标
# print(res.end())  # 结束下标
#
# res = re.match('com', 'python.org')
# print(res) # 找不到返回None
#
#
# # 用正则表达式模式去查找
# line = 'Cats are smarter than dogs'
# # .* 任意字符（不包含换行）任意个数
# # re.M 多行匹配模式（默认是单行）
# # re.I 忽略大小写（默认区分大小写）
# res = re.match(r'.* are .*', line, re.M | re.I)
# print(res)
# # print(res.group()) # 返回匹配到的字符


res = re.search('hqyj', 'www.hqyj.com')
print(res)
print(res.group())

s = '321fjdkafl8098fjdslajfl^*^*^*'
res = re.search(r'\d+', s)
print(res.group())
res = re.findall(r'\d+', s)
print(res)


