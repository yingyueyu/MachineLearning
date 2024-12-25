# pip install pandas
import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, 8]) # int64类型
# s = pd.Series([1, 3, 5, 8, np.nan]) # int64类型
# s = pd.Series([1, 3, 5, np.nan, 8, 3.14])  # float64
# s = pd.Series([1, 3, 5, np.nan, 8, 3.14, 'a'])  #object
print(s)

# 创建一个日期的序列，返回DatetimeIndex
dates = pd.date_range('20240701', periods=7)
print(dates)

# index和columns参数可以省略，如果省略不写，自动生成数字序列作为index或标题
# 如果设置了index和columns， index的长度跟数据的行数一致； columns的数量也要跟数据列数相同
df1 = pd.DataFrame(np.random.randn(7, 5), index=dates, columns=list('ABCDE'))
print(df1)

# 用字典作为参数生成dataframe， key就是标题， value是这一列的内容.
# 每一列长度相同或者可以广播为相同
df2 = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a','b','c','d','e'],
    'C': 1,
    'D': pd.Timestamp('20240724'),
    'E': pd.Series(np.arange(5)),
    'F': np.arange(1, 6),
    'G': 'aaa',
    'H': pd.Categorical(['男','女','女','男','女'])
})
print(df2)


# 查看数据
print(df1.head(3)) # 返回前面3行数据
print(df1.tail(3))  # 返回最后三行数据
print(df1.index)  # 返回所有索引
print(df1.columns)  # 返回所有标题
print(df1.describe())  # 返回每一列的数量，平均值，最大最小，标准差等信息

print(df1)
# axis=0, 按行排序； axis=1，按列排序
print(df1.sort_index(axis=1, ascending=False))

# 按照指定列的值进行排序
print(df1.sort_values(by='A'))

# DateFrame对象转Numpy数组。 只会转数据部分，索引不会转
a = df1.to_numpy()
print(a)
print(df1)

# 获取数据
#  返回指定列的数据, 参数是列名（标题名）
print(df1['A'])
print(df1.A)  #  返回列数据简化写法

# 切片方式返回行
print(df1[0: 3])  #  用序号切片
print(df1['20240702': '20240704'])  # 用索引切片， 包含结束位置

# 用行的索引返回一列数据，返回的类型为Series
print(df1.loc['20240701'])
# 返回指定行列的单元格的数据
print(df1.loc['20240701', 'A'])
# 切片访问，不能用序号，用索引
print(df1.loc['20240701': '20240703', 'A': 'C'])
print(df1.loc['20240701': '20240703', ['A', 'C']])

# 按位置获取数据
print(df1.iloc[3])  #  返回行号为3的这一行的数据
print(df1.iloc[0: 2, 1: 3])  # 返回从0到2行， 从1到3列这个区间内的数据（区间左闭右开）
print(df1.iloc[..., 2: 4])
print(df1.iloc[0:3, :])
print(df1.iloc[1, 1])
df1.iloc[1, 1] = 888
print(df1)
#  增加一行数据
df1.loc[pd.Timestamp('20240708')] = [0, 0, 0, 0, 0]
#  新增一列数据
df1['F'] = 1
print(df1)

# 按条件选择
print(df1[df1.A > 0])  #  返回A列大于0的所有行（整行）
print(df1[df1 > 0])  # 返回表格完整的形状，大于0原样返回数据，小于0用NaN填充

# 赋值
# 整列赋值. 赋值一个Series对象， Series要指定index
df1['A'] = pd.Series(np.arange(8), index=df1.index)
print(df1)
# 单元格赋值
df1.loc['20240702', 'B'] = 999
df1.iloc[1, 1] = 666
# 给切片出来的矩阵赋值， 赋值的形状要跟切片形状相同
df1.iloc[2: 3, 2: 4] = np.array([555, 777]).reshape(1, 2)
# 根据条件赋值
df1[df1 < 0] = np.abs(df1)
# 删除列
# del df1['F']
# 删除列。 返回删除后的DataFrame
print(df1.drop('F', axis=1))
# 删除行。 返回删除后的DataFrame
print(df1.drop('20240703', axis=0))
print(df1)
#  增加一行数据
df1.loc[pd.Timestamp('20240708')] = [0, 0, 0, 0, 0, 0]
#  新增一列数据
df1['F'] = '壹'
print(df1)

# 文件操作
# 把DataFrame保存为excel文件
# 如果环境没有安装 openpyxl的库，需要安装： pip install openpyxl
# df1.to_excel('demo.xlsx', sheet_name='测试页')



