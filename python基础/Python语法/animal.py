# 父类
class Animal:

    def __init__(self):
        print('Animal的init')

    def run(self):
        print('动物正在跑...')

    def sleep(self):
        print('动物正在睡觉...')

    def eat(self):
        print('动物正在吃东西...')


# 子类， 继承Animal
class Cat(Animal):

    def __init__(self):
        print('Cat的init')
        # 调用父类的构造方法
        # super(Cat, self).__init__()
        Animal.__init__(self)

    def __str__(self):
        return '这是一只猫'

    def catch_mouse(self):
        # 子类的方法中调用父类的方法。 类名.方法名(self)
        Animal.run(self)
        print('猫抓老鼠...')

    def sleep(self):
        print('猫在睡觉...')


class Parent1:

    def method1(self):
        print('父类1的方法')


class Parent2:
    def method2(self):
        print('父类2的方法')


class Child(Parent1, Parent2):
    pass


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 重写str方法
    def __str__(self):
        return f'Vector: {self.x}, {self.y}'

    # 重写+运算符的内置方法
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)


class Counter:
    # 类的私有属性. 只能在类的内部用类名.属性名或self.属性名方式访问
    __count = 0

    def count(self):
        self.__plus()
        # print(f'计数器的结果： {Counter.__count}')
        print(f'计数器的结果： {self.__count}')

    def __plus(self):
        self.__count += 1