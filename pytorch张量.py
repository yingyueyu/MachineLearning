# 知识点
# 什么是张量
#   张量是 pytorch 运算的基本单元，pytorch 运算时是 张量 与 张量 进行运算，张量中存储的其实是多维数据

# 重点:
# 1. 张量创建
# 2. 张量访问
# 3. 张量形状修改
# 4. 矩阵运算

import math

import torch

# 创建空张量
t = torch.empty(2, 3, 4)
print(t)
print(t.shape)

# 查询类型
print(type(t))
# 张量内的数据类型
print(t.dtype)

# 创建 0 张量
t = torch.zeros(2, 3, 4)
print(t)
print(t.shape)

# empty  vs  zero
# empty: 创建张量但不初始化
# zero: 创建张量并初始化为 0


# 指定数据类型
t = torch.zeros(2, 3, 4, dtype=torch.int)
print(t)
print(t.dtype)

# 转换数据类型
t = t.to(torch.float)
print(t.dtype)

# 创建 1 张量
t = torch.ones(2, 3)
print(t)
print(t.shape)

# 随机张量
# 设置随机种子
torch.manual_seed(100)

t = torch.rand(2, 3)  # 随机 0~1
print(t)
t = torch.randn(2, 3)  # 随机正态分布的值
print(t)
t = torch.randint(2, 10, (3, 4))  # 随机整数
print(t)
t = torch.normal(mean=0, std=1, size=(2, 3))
print(t)

# 通过以下 api 创建形状和 t 相同的张量
t = torch.empty_like(t)
t = torch.zeros_like(t)
t = torch.ones_like(t)
t = torch.rand_like(t.to(torch.float))
t = torch.randn_like(t)
t = torch.randint_like(t, 2, 10)
print(t)

# 访问张量成员
t = torch.arange(10).view(2, 5)
print(t)
print(t[1, 2])  # 结果依然是张量
print(t[:, 1:4])  # 切片

# 获取张量内数据
print(t[1, 2].item())

# 查看形状
print(t.shape)
print(t.size())

# 修改形状
# 注意，参数的乘积，必须等于原形状各维度的乘积
t = torch.ones(3, 4)
# 获取张量中的元素数量
print(t.numel())

# reshape
t = t.reshape(2, 6)
print(t.shape)

# view
t = t.view(1, 2, 6)
print(t.shape)

# reshape vs view
# reshape: 重塑形状是内存会重新分配
# view: 重塑形状是内存不会重新分配

# permute: 改变维度顺序
t = t.permute(2, 0, 1)
print(t.shape)

# transpose: 转置，交换两个维度
t = t.transpose(1, 0)
print(t.shape)

# squeeze 挤压
x = torch.ones(2, 1, 3, 1, 3)
# 默认将长度为1的所有维度去掉
print(torch.squeeze(x).shape)

# 指定维度的长度为1则去掉，否则保留
print(torch.squeeze(x, 3).shape)

# 实例方法
print(x.squeeze(2).shape)
print(x.squeeze(1).shape)

# unsqueeze 放松
x = torch.ones(2, 3)
# 在指定位置的维度上，新增长度为1的维度
print(torch.unsqueeze(x, 0).shape)
print(torch.unsqueeze(x, 1).shape)
print(torch.unsqueeze(x, 2).shape)

# expand 扩展
x = torch.ones(1, 3)
# 将某个长度为 1 的维度扩展为指定长度
print(x.expand(4, 3).shape)

# 手动设置张量值
t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float, device='cpu')
print(t)

# 和标量运算
print(3 + t)
print(t - 1)
print(t / 2)
print(t // 2)
print(t % 2)
print(t ** 2)

# 具有相同形状的张量可以进行逐位的数学运算
a = torch.tensor([
    [1, 2, 3],
    [0, 1, 2]
])
b = torch.tensor([
    [2, 1, 0],
    [0, 1, 1]
])
print(a + b)
print(a * b)
print(a ** b)

# 张量广播
# 什么是张量广播?
# PyTorch 中的张量广播（broadcasting）是一种机制，用于在执行元素级操作时自动扩展具有不同形状的张量，使它们具有兼容的形状。这使你能够执行一些操作，而无需显式地扩展张量的形状。

# 张量广播条件:

# 1. 维度数量相同时，对于每个维度，两个张量的大小要么相等，要么其中一个张量的大小为1。

# 2x2x3
x = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# 2x1x3
y = torch.tensor([[[1, 2, 1]], [[3, 2, 3]]])
# print(x + y)
# print(x + y.expand_as(x))
print(x + y.expand(-1, 2, -1))

# 2. 如果两个张量的维度数不同，将较小的张量的形状在其前面补1，直到维度数相同。

# 2x2x3
x = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# 1x2x3
y = torch.tensor([[1, 2, 3], [3, 4, 2]])
# print(x + y)
# print(x + y.unsqueeze(0).expand(2, -1, -1))

# # 三角函数
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print(angles)
print(sines)
print(inverses)

# 判断张量各位置是否相等
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[1, 2], [4, 4]])
# 两个张量内容是否完全一致
print(torch.equal(x, y))

# 两张量逐个位置判断是否相等
# 这种判断相等的作用，用于获取bool矩阵掩码，用于对后续内容进行操作
print(torch.eq(x, y))
print(x == y)  # 等价于 torch.eq(x, y)
x = torch.arange(10)
print(x)
print(x < 5)
x[x < 5] = 0  # 使用掩码取值或赋值
print(x)

# 是否近似相当
# 作用: 常用于判断两个张量是否数值相当
x = torch.tensor([[1.0008, 2], [4, 4]])
y = torch.tensor([[1, 2.0007], [4, 4]])
print(torch.allclose(x, y, atol=1e-3))

# 张量中最大值
x = torch.arange(10).view(2, 5)
print(x)
print(torch.max(x, dim=1))
print(torch.max(x, dim=0))

# 均值
# 求指定dim维度上的均值
x = torch.tensor([
    [
        [
            [0.6677, 0.5046],
            [0.3094, 0.9748]
        ],

        [
            [0.1465, 0.4524],
            [0.6125, 0.4492]
        ]
    ],

    [
        [
            [0.8902, 0.0705],
            [0.4210, 0.7535]
        ],

        [
            [0.5091, 0.0504],
            [0.3518, 0.7547]
        ]
    ]
])

# (2,2,2,2)
print(x.shape)

# 在不同维度上，值对应位置相加再除以批次数
# dim=0，代表最外层方括号，里面有2个批次
# [[[0.6677, 0.5046],[0.3094, 0.9748]],[[0.1465, 0.4524],[0.6125, 0.4492]]]
# [[[0.8902, 0.0705],[0.4210, 0.7535]],[[0.5091, 0.0504],[0.3518, 0.7547]]]
# 然后对位相加求平均数
print(torch.mean(x, dim=0))
# dim=1，代表从外至里的第二层方括号，这里有两组，第一组为
# [[[0.6677, 0.5046],[0.3094, 0.9748]],[[0.1465, 0.4524],[0.6125, 0.4492]]]
# 然后第一组内，元素对位相加除以批次数，例如
# (0.6677 + 0.1465) / 2, (0.5046 + 0.4524) / 2
# (0.3094 + 0.6125) / 2, (0.9748 + 0.4492) / 2
print(torch.mean(x, dim=1))
# dim=2，代表从外至里第三层方括号，例如，第一组: [[0.6677, 0.5046],[0.3094, 0.9748]]
# 然后对位相加，再除以批次数
# (0.6677 + 0.3094) / 2, (0.5046 + 0.9748) / 2
print(torch.mean(x, dim=2))
# dim=3, 代表最里面的括号，那么平均数就是求这个列表内所有成员和除以个数，例如第一行:
# (0.6677 + 0.5046) / 2
print(torch.mean(x, dim=3))
# print(torch.mean(x, dim=4))


# 方差
# 计算步骤
# 先求 mean 平均值
# 再求每个元素和平均值的平方差
# 再将对应维度相加
print('方差')
print(torch.var(x, dim=0))
# 拆解步骤:
# 1. 先求 mean 平均值
r = torch.mean(x, dim=0)
print(r)
# 2. 再求每个元素和平均值的平方差
r = (x - r) ** 2
print(r)
# 3. 再将对应维度相加
r = torch.sum(r, dim=0)
print(r)

# 标准差
print('标准差')
print(torch.std(x, dim=0))
# 步骤解析:
# 在方差基础上，开平方根得到
r = torch.var(x, dim=0)
r = torch.sqrt(r)
print(r)

# 矩阵乘法
# A 是 3x2 那么 B 就必须是 2x3 否则不能相乘
# 相乘结果形状是 3x3; 取 A 的行数和 B 的列数
# 矩阵乘法算法:
# A 的第一行点乘 B 的第一列并求和
# A 的第二行点乘 B 的第一列并求和
# A 的第三行点乘 B 的第一列并求和
# 前三次运算结果放入新矩阵的第一列
# 再次用 A 的每行和 B 的第二列点乘并求和；结果放入新矩阵第二列
# 再次用 A 的每行和 B 的第三列点乘并求和；结果放入新矩阵第二列

# 批量矩阵乘法
# 结论:
# 1. A B 形状的右上到左下对角线相等则可以相乘
# 2. A B 相乘结果 C 的形状为 A B 形状的左上和右下
# 3. 矩阵相乘不具备交换性
A = torch.randint(0, 5, (2, 3))
B = torch.randint(0, 5, (5, 3, 4))
print(A)
print(B)
C = torch.matmul(A, B)  # 会张量广播
print(C)
print(C.shape)

C = torch.mm(A, B)  # 不会张量广播
print(C)

# 批量矩阵相乘 batch matmul
A = A.unsqueeze(0).expand(5, -1, -1)
B = B.unsqueeze(0).expand(5, -1, -1)
print(A.shape)
print(B.shape)
print(torch.bmm(A, B).shape)

# 克隆
x = torch.ones(2, 2)
# y = x
y = x.clone()
print(torch.equal(x, y))
x[0, 0] = 2
print(x)
print(y)
