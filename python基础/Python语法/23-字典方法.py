# 数字(整数和小数)作为key
dict1 = {1: '求知星', 2: '卓越星', 3: '未来星'}

dict2 = {'name': '张三', 'age': 22, 'city': 'guiyang', 'name': '小明'}
# 字典的访问用key作为下标，不能用序号做下标
print(dict2['age'])
# 如果字典里有相同的key， 会用后面的key覆盖前面的key
print(dict2['name'])

# 用元组作为key（因为元组是不可改变的，可以作为key）
dict3 = {(1, 2): 'value1', ('a', 'b'): 'value2'}
print(dict3[(1, 2)])

dict4 = {}
print(len(dict4))

dict5 = dict()
print(type(dict5))

# list1 = list()
# print(type(list1))
#
# tup1 = tuple()
# print(type(tup1))

# 访问不存在的key会报异常
# print(dict1[4])

# ==========修改字典===========
# 对指定的key重新赋值，就是修改
dict1[1] = ['求知星5G', '求知星2.4G']
print(dict1[1])
# 对一个不存在的key进行赋值，相当于是新增
dict1[4] = '远航星'
print(len(dict1))

# del 删除指定的key， 键值对一起被删掉
del dict1[2]
print(dict1)
# 如果直接删字典变量名，删除的是变量
# del dict1

# ========dict方法=============
print(dict1)
# for循环遍历得到的是key
for d in dict1:
    print(d)

# 返回字典的key的列表
keys = dict1.keys()
print(type(keys))
print(keys)
for k in keys:
    print(k)

# 返回字典里面的value的列表
values = dict1.values()
print(type(values))
print(values)

# 返回键值对的列表，键值对以元组形式表示的：[(1, ['求知星5G', '求知星2.4G']), (3, '未来星'), (4, '远航星')]
items = dict1.items()
print(type(items))
print(items)

# 通过key读取value. 如果key不存在，返回None
print(dict1.get(4))
print(dict1.get(5))

# 复制一个新的字典
dictNew = dict1.copy()
print(dictNew)

# 合并两个字典
dict1.update(dict2)
print(dict1)

# 根据列表的内容生成字典的key, value默认为None， 可以通过第二个参数指定value
dictNew = dict.fromkeys(['a', 'b', 'c'])
print(dictNew)

# 为key设置默认值， 在get时如果取不到key则返回它对应的默认值
dict1.setdefault(5, '888')
print(dict1.get(5))

# 返回指定key对应的value， 同时会删除字典里这个key所对应的键值对
print(dict1.pop('city'))
print(dict1)

# 删除并返回字典最后一个键值对，以元组的形式返回
print(dict1.popitem())
print(dict1)

# 清空字典
dict1.clear()
print(dict1)




