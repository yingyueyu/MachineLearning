
# 位置参数
def showStu(name, age, sex):
    print(f'姓名： {name}, 年龄： {age}, 性别： {sex}')


showStu('珊珊', 21, '女')
# 参数数量不对，报异常
# showStu('小明')
#  参数顺序不低，导致逻辑错误
showStu('男', '小张', 22)



# 默认值函数
# 参数名=默认值  这样的格式给参数提供默认值， 如果该参数不传，就用默认值代替
def showStu2(name, age=20, sex='男'):
    print(f'姓名： {name}, 年龄： {age}, 性别： {sex}')


showStu2('小明')


# 关键字参数。
# 调用函数时指定参数名， 这样就可以不用按形参顺序传递参数
showStu(sex ='男', name ='小张', age=22)


# 课堂练习：定义一个函数，计算bmi = 体重（kg）/ 身高（m） ** 2
def bmi(height, weight):
    if height <= 0:
        return '身高不正确'

    result = ''
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        result = "偏瘦"
    elif bmi < 25:
        result = "正常"
    else:
        result = "偏胖"
    return result


print(f'你的Bmi指标为：{bmi(height=1.8, weight=90)}')


# 函数返回值
# 返回100~999之间的水仙花数(每一位数字的立方和等于数字本身)
def daff():
    result = []  # 存水仙花数的列表
    for i in range(100, 1000):
        # a = i // 100
        # b = (i % 100) // 10
        # c = i % 10
        s = str(i)
        a = int(s[0])
        b = int(s[1])
        c = int(s[2])
        if (a ** 3 + b ** 3 + c ** 3) == i:
            result.append(i)
    return result  # 返回list类型


l = daff()
print(l)
print(type(l))


# 用一个函数计算1~100的总和，奇数和， 偶数和
def tjo():
    # 定义三个和的变量
    total = 0
    totalJi = 0
    totalOu = 0
    # 循环 1~100
    for i in range(1, 101):
         #累加
        total += i
         #if 奇数
        if i % 2 != 0:
              #奇数和累加
            totalJi += i
         #else  偶数
        else:
             # 偶数和累加
            totalOu += i
    return total, totalJi, totalOu


# 函数返回多个值， 用对应数量的变量接收它的返回
x, y, z  = tjo()
print(f'总和： {x}, 奇数和： {y}, 偶数和： {z}')
tup1 = tjo()
print(tup1, type(tup1))
print(tup1[0])

x = 5050
result = 0
# 减法函数
def dec(a, b):
    global result  #  函数内部要修改全局变量，需要用关键字global来声明
    result = a - b
    print(result)
    x = 500 #  函数内无法修改全局变量的
    print(x)


dec(8, 5)
print(x)
print(result)


