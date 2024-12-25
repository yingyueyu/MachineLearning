def Fibonacci(n):
    if n <= 0:
        raise BaseException('数列的序号必须从1开始')
    if n < 3:
        return 1
    else:
        return Fibonacci(n - 1) + Fibonacci(n - 2)


print(Fibonacci(int(input('请输入佩波那契数列的项数：'))))


class Student:
    def __init__(self, name, age, sex, phone):
        self.name = name
        self.age = age
        self.sex = sex
        self.phone = phone

    def study(self):
        print(f'{self.name}正在学习搬砖...')

    def play(self):
        print(f'{self.name}玩缝纫机...')

    def call(self):
        print(f'{self.name}号码是{self.phone}...')

    def age(self):
        print(f'{self.name}年龄{self.age}')

    def sex(self):
        print(f'{self.name}性别是{self.sex}')


# from com.hqyj.stu.Student import Student
stu1 = Student('吴某凡', 30, '男', 123456)
stu2 = Student('张张', 20, '男', 456789)
stu1.study()
stu1.play()
stu1.call()
print(f"{stu1.name} sex is {stu1.sex}")
print(f"{stu1.name} age is {stu1.age}")
stu2.study()
stu2.play()
stu2.call()
print(f"{stu2.name} sex is {stu2.sex}")
