import torch
from torch import nn

# 什么是就地操作 in-place？
# 直接改变原始张量的操作

# 限制: 不能在叶节点使用 in-place

A = torch.tensor([-1, -2, 1, 2])
# B = A + 1
relu = nn.ReLU(inplace=True)
print(relu(A))
print(A)
