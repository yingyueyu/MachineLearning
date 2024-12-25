# import numpy as np
# a = np.array([
#     [
#         ['a', 'b'],
#     	['c', 'd']
#     ],
#     [
#         ['e', 'f'],
#     	['g', 'h']
#     ]
# ])
# b = np.rollaxis(a, 2, 1)
# print(b)

import pandas as pd

# df1 = pd.DataFrame([['张三', '男', 20], ['李四', '男', 18], ['王芳', '女', 22]], columns=['姓名', '性别', '年龄'])
# df1.index.name = '学号'
# df1.loc[3] = ['王瑶', '男', 22]
# df1.loc[4] = ['林洛瑶', '女', 18]
# df1.to_excel('students.xlsx', sheet_name='1班')

df2 = pd.read_excel('students.xlsx', '1班', index_col='学号')
# 先用条件筛选出符合条件的DataFram, 遍历显示
p = df2[df2['姓名'].str.endswith('瑶')]
for i in p.index:
    print(p.loc[i, '姓名'], p.loc[i, '年龄'])

## 直接遍历所有行，判断姓名是否符合要求
# for i in df2.index:
#     if df2.loc[i, '姓名'].endswith('瑶'):
#         print(df2.iloc[i, 0], df2.iloc[i, 2])