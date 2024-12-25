#============定义和访问==========
set1 = {3.14, 1, 2, 3, 4, 5, 'aaa', '7ew78'}
set2 = set(['a', 'b', 'c'])

# set不能用下标方式访问
# print(set1[2])

for i in set1:
    print(i)

# 新增元素。 集合的元素可以是任意类型
set1.add('xyz')
print(set1)

# 删除指定的元素， 如果不存在报异常
set1.remove('xyz')
print(set1)
# set1.remove('abc')

# 删除指定的元素，如果不存在不会抛异常
set1.discard(5)
print(set1)
set1.discard(8)

# 删除集合最左边的一个元素并返回， 但是集合排序是不确定的，所以无法确定会删哪一个元素
print(set1)
print(set1.pop())
print(set1)

# 合并集合
set1.update(set2)
print(set1)

set3 = {1, 2, 3, 4, 5, 6, 7}
set4 = {1, 9, 3, 8, 5, 10, 11}

# 返回set3中存在， set4不存在的元素
setDiff = set3.difference(set4)
print(setDiff)  # {2, 4, 6, 7}

# 删除set3中两个集合的交集
set3.difference_update(set4)
print(set3)

# 返回交集（两个集合都有的元素）
setInter = set3.intersection(set4)
print(setInter)
# 删除set3的非交集元素（只保留交集）
set3.intersection_update(set4)
print(set3)

# 返回并集（两个集合合并到一起，相同的元素只出现一次）
setUni = set3.union(set4)
print(setUni)

# 返回两个集合中没有交集的所有元素
setSd = set3.symmetric_difference(set4)
print(setSd)
# set3更新为两个集合没有交集的所有元素，交集元素会去掉
set3.symmetric_difference_update(set4)
print(set3)

set5 = {1, 2, 3, 4, 5, 6}
set6 = {2, 3, 5}
# set6是不是set5的子集
print(set6.issubset(set5))
# set5是否是set6的超集
print(set5.issuperset(set6))
# 判断两个集合是否没有交集，没有返回True，有交集返回False
print(set5.isdisjoint(set6))








