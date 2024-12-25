# 先创建员工类Employee

from Employee import Employee

emp1 = Employee('张三', 28)
emp1.work()

emp2 = Employee('李四', 30)
emp2.work()

emp3 = Employee('王五', 40)


Employee.emp_count += 1

print(emp1.emp_count)
print(emp2.emp_count)
print(Employee.emp_count)


print(emp1.age)
print(emp1.name)
emp1.age = 29 # 修改属性
print(emp1.age)

# 通过对象添加删除属性, 只能对当前对象有效
# emp1.score = 100
# print(emp1.score)
# print(emp2.score)
# del emp1.score #  删除对象的属性
# print(emp1.score)

# 属性操作的方法
# 是否有指定的属性
print(hasattr(emp1, 'score'))
# 返回对象的属性
print(getattr(emp1, 'age'))
# 修改属性的值
setattr(emp1, 'age', 35)
# 删除属性， 如果属性不存在，会报异常
# delattr(emp1, 'score')



# =======类的内置属性=========
print('类的文档注释:', Employee.__doc__)
print('类名：', Employee.__name__)
print('类定义所在的模块：', Employee.__module__)
print('类的所有父类：', Employee.__bases__)
print('类的所有信息：', Employee.__dict__)



