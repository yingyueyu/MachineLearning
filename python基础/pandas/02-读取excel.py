import pandas as pd

# 读取excel文件，返回DataFrame。 index_col用于指定索引列的序号，如果不设参数，自动生成0开始的数字序号的索引列
df3 = pd.read_excel('demo.xlsx', '测试页', index_col=0)
print(df3)