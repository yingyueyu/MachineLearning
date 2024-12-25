import torch
from torch import nn

a = torch.arange(0, 9).reshape(3, 3)
print(a)
a = torch.transpose(a, dim0=1, dim1=1)
print(a)

cont = nn.ConvTranspose2d(12, 6, 3, 1)
image = torch.randn(1, 12, 3, 3)
# 计算公式  假设起始的目标大小为n，卷积大小为m 步长为s，卷积核大小为k 边距大小为p
#  (n - 1) * s - 2p + m
# 512 = 15 * 32 - 2p + 32
result = cont(image)
print(result.shape)
