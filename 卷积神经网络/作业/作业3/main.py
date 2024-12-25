# - 创建输入数据x，形状为 (5, 3, 100, 100)
# - 创建一个3x3卷积conv3，输出3通道
# - 创建一个7x7卷积conv7，输出3通道
# - 使用不同的卷积，卷积输入数据x:
#   1. 使用数据x经过三次conv3，打印输出形状
#   2. 使用数据x经过一次conv7，打印输出形状
# - 打印上述两种过程中，参与运算的权重数


import torch
from torch import nn

x = torch.randn(5, 3, 100, 100)

conv3 = nn.Conv2d(3, 3, 3, bias=False)
conv7 = nn.Conv2d(3, 3, 7, bias=False)

print(conv3(conv3(conv3(x))).shape)
print(conv7(x).shape)

print(sum([p.numel() for p in conv3.parameters()]) * 3)
print(sum([p.numel() for p in conv7.parameters()]))

print(conv3.weight.numel() * 3)
print(conv7.weight.numel())
