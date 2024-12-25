import re
# 写文件
f1 = open('data.txt', 'w', encoding='utf-8')
f1.write('帝高阳之苗裔兮，朕皇考曰伯庸。\n'
         '摄提贞于孟陬兮，惟庚寅吾以降。\n'
         '皇览揆余初度兮，肇锡余以嘉名。\n'
         '名余曰正则兮，字余曰灵均。\n'
         '纷吾既有此内美兮，又重之以修能。\n'
         '扈江离与辟芷兮，纫秋兰以为佩。\n'
         '汩余若将不及兮，恐年岁之不吾与。\n'
         '朝搴阰之木兰兮，夕揽洲之宿莽。\n'
         '帝高阳之苗裔兮，朕皇考曰伯庸。\n'
         '帝高阳之苗裔兮，朕皇考曰伯庸。\n'
         '帝高阳之苗裔兮，朕皇考曰伯庸。\n'
         '帝高阳之苗裔兮，朕皇考曰伯庸。\n'
         '高阳之苗裔兮，朕皇考曰伯帝ad234;庸。\n'
         '日月忽其不淹兮，春与秋其代序。')
# f1.seek(0)
f1.close()

# 读文件
with open("data.txt", 'r', encoding='utf-8') as f2:
    # 读取所有行
    lines = f2.readlines()
    # print(lines)
    # 把返回的字符串列表拼接为一个字符串
    s1 = ','.join(lines)
    # print(s1)

# 用正则表达式查找'帝高阳之苗裔兮，朕皇考曰伯庸。\n'
l1 = re.findall(r'帝.*庸', s1, re.M)
# print(l1)



# class FindStr:
#     def __init__(self, find_str):
#         self.find_str = find_str
#         self.i = -1
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         self.i += 1
#         if self.i < len(self.find_str):
#             return self.find_str[self.i]
#         else:
#             raise StopIteration

def getstr():
    for s in l1:
        yield s



print('找出所有指定字符串的索引为：')
# for values in FindStr(l1):
#     print(values)

for values in getstr():
    print(values)
