stus = ['刘勇', '罗永欢', '杨豪']

# 通过for循环把列表里的每个元素依次取出来赋给变量stu
for stu in stus:
    print(stu)

# 利用range产生循环，累加1到100的和
sum = 0
for i in range(1, 101):
    sum += i
print(sum)

# tuple也可用于for循环迭代
t1 = ('C++', 'Java', 'SQL', 'Python')
for lan in t1:
    print(lan)


# 字典用于循环迭代
dics = {'张三': 98, 'lisa': 60, (2, 3): 6}
for dic in dics:
    print(dic) # 字典循环只能取出key
    print(dics[dic])  # 如果要取value，要通过key做为下标取取


'''
打印如下效果

张三： 唱跳  打篮球
小杨： 历史  心理  篮球
'''
hobby = {'张三': ['唱跳','打篮球'], '小杨': ['历史', '心理', '篮球']}
for ho in hobby: # 外层循环，循环每一位同学
    print(ho, end=': ') # 不换行，每个学生后面打冒号
    values = hobby[ho] # 当前这位同学的爱好列表
    for v in values: # 内层循环，循环每一位同学的所有的爱好
        print(v, end="  ") # 不换行，每个爱好后面打印空格
    print('') # 打印换行


# 爱好之间用逗号分隔
hobby = {'张三': ['唱跳', '打篮球'], '小杨': ['历史', '心理', '篮球']}
for i in hobby:
    print(i, end=': ')
    hobbies_list = hobby[i]
    hobbies_last = hobbies_list[len(hobbies_list)-1] # 读取最后一个元素
    for j in hobbies_list:
        if j != hobbies_last:
            print(j, end=',')
        else:
            print(j, end='。')
    print()


# 循环读取集合的元素
set1 = {'人工智能', '嵌入式', '机器人'}
for course in set1:
    print(course)