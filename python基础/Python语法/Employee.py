class Employee:
    """
    员工类，描述员工的信息
    """

    # 类的属性
    emp_count = 0

    # 类的构造方法，创建对象时要执行的方法
    # 初始化信息放在构造方法的参数传进来
    def __init__(self, name=None, age=None):
        self.name = name
        self.age = age
        Employee.emp_count += 1

    def work(self):
        print(f'{self.name}正在努力工作...')

    def rest(self):
        print('正在午休...')