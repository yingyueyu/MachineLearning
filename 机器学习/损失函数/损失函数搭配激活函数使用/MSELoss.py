# MSELoss 均方误差损失函数，多用于回归问题，衡量样本和预测结果的差距
# 应用场景: 多用于回归问题求损失

import torch
import torch.nn as nn

# 输入数据，假设3个样本，每个样本有10个特征
x = torch.randn(3, 10)
# 真实标签
label = torch.tensor([1, 2, 3]).view(3, -1)

# 全连接神经网络
fc = nn.Linear(10, 1)

y = fc(x)

# 声明损失函数
loss_fn = nn.MSELoss()

# 第一个参数是模型预测值，第二个是真实值
loss = loss_fn(y, label)

print(loss)
