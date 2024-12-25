# import animal
# import语句一次可以导入多个对象
from animal import Cat, Child, Parent2, Parent1, Animal, Vector, Counter

# cat = animal.Cat()
cat = Cat()
cat.catch_mouse() # 子类自己的方法
cat.run()  # 父类的方法
cat.sleep() # 父类的方法
cat.eat() # 父类的方法
print(cat)


# 创建多继承子类的对象
child = Child()
child.method1()
child.method2()

# id() 方法可以查看对象的id， 不同的对象id完全不一样
print(id(child))
child2 = Child()
print(id(child2))
child3 = child2
print(id(child3))
print(child3 == child2)  #  True

# 返回一个类是不是另一个类的子类（或子孙类）
print(issubclass(Child, Parent2))

# 返回对象是否是类的实例
print(isinstance(cat, Animal))



#  +运算符重写
v1 = Vector(5, 8)
v2 = Vector(2, -3)
print(v1 + v2)


counter = Counter()
counter.count()
counter.count()
# 类的私有属性不能在类外面访问

try:
    # print(counter._Counter__count)
    # print(Counter.__count)
    a = 1 / 0
except AttributeError:  # 出现异常进入该分支
    print('该属性不存在')
# else:   # 正常执行进入该分支
#     print('正常执行进入该分支')
# finally:
#     print('始终会执行的分支')


# 类的私有属性可以通过对象._类名私有属性名这样的方式访问
print(counter._Counter__count)