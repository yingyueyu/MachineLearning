import torch

# 1. 创建一个有数字 1 ~ 20 共 20 个数字的张量，且形成为 (4, 5) 的矩阵，该矩阵命名为 A
A = torch.arange(1, 21).view(4, 5).to(torch.float)
print(A)

# 2. 选择 A 中 小于 7 的部分数据将值设置为 float('-inf')

# where = A < 7
mask = A < 7
# A[mask] = float('-inf')
A = A.masked_fill(mask, float('-inf'))
print(A)

# 3. 随机一个张量，张量元素是 -3 ~ 10 之间的浮点数，张量形状为 (4, 5)，命名为 B
# 下限 + 随机数 * (上限 - 下限)

B = -3 + torch.rand(4, 5) * (10 - (-3))
print(B)

# 4. 计算 $A^TB$

print(torch.matmul(A.transpose(0, 1), B))
# 简写
# 1. 转置的简写 .T
# 2. @ 表示矩阵乘法 等价于 torch.matmul 还等价于 torch.bmm
_A = A.T.unsqueeze(0).expand(5, -1, -1)
print(_A.shape)
print(_A @ B)

# 5. 将 B 扩展成 (3, 4, 5) 的形状

B = B.unsqueeze(0).expand(3, -1, -1)
print(B.shape)

# 6. 再次计算 $A^TB$

print((A.T @ B).shape)

# 7. 将 A 转置并扩展成 (3, 5, 4) 的形状

A = A.T.unsqueeze(0).expand(3, -1, -1)
print(A.shape)

# 8. 再次计算 $AB$

print(torch.bmm(A, B).shape)
# print((A @ B).shape)
