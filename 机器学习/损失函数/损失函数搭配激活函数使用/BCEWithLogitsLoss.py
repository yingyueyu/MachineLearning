# BCEWithLogitsLoss 包含了 sigmoid 的 BCELoss

import torch
from torch import nn

torch.manual_seed(100)

# 假设三个分数，60分及格
x = torch.tensor([[80, 60, 40]]).T.to(torch.float)
# 1: 及格， 0: 不及格
label = torch.tensor([[1, 1, 0]]).T.to(torch.float)

# 通过全连接学习更多的特征，然后再进行分类
fc1 = nn.Linear(1, 10)
fc2 = nn.Linear(10, 1)

y = fc2(fc1(x))
# 激活
# nn.functional.sigmoid
# sigmoid = nn.Sigmoid()
# y = sigmoid(y)

# 求损失
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.BCELoss()
loss = loss_fn(y, label)
print(loss)
