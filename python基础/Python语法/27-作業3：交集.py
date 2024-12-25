# 定义两个集合，分别是两个同学选修课的集合。找出两位同学公共选修的课程

stu1 = {'c#', 'python', 'java', 'c++', 'unity'}
stu2 = {'javascript', 'python', 'java', 'mysql', 'php'}
courses = stu1.intersection(stu2)
print(f'公共课课程：{courses}')