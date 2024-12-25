import torch
from torch import nn

# a = torch.randn(1, 1, 3, 3)
a = torch.randn(1, 1, 28, 28)
# layer = nn.ConvTranspose2d(1, 1, 3,1, 0, dilation=2)
# 向上采样
layer = nn.ConvTranspose2d(1, 1, 3,1, 12, dilation=12)
# 向下采样
conv_layer = nn.Conv2d(1,1,3,1,1,dilation=12)
# print(b.shape)
