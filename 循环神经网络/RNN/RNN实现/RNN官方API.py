import torch
import torch.nn as nn

# 官方RNNCell
cell = nn.RNNCell(2, 10)

# 官方RNN
# batch_first: 是否将批次放到第一个维度
model = nn.RNN(2, 10, batch_first=False)
# 序列形状: (L, N, input_size)
s1 = torch.randn(3, 5, 2)
y, h = model(s1)
print(y.shape)
print(h.shape)
