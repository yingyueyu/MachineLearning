# 什么是多分类问题?
# 多分类问题是指有多个类别的问题，每个样本可以属于其中的一个类别。例如，在图像分类问题中，每个图像可以属于多个类别，如“猫”、“狗”、“鸟”等。
# 这里我们就以 猫狗鸟 三分类为例，0 代表猫，1 代表狗，2 代表鸟。
import random

import torch
from torch import nn

# 随机图片
x = torch.rand(64, 100, 100)
# 随机标签
y = torch.randint(0, 3, (64,))


# 声明一个pytorch的自定义模块
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential: 这是一个序列，用于依次执行序列中的层(模块)
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10000, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=-1)
        )

    # 前向传播: 模型预测和推理的过程
    def forward(self, x):
        return self.stack(x)


# 初始化模型
model = Model()

# 损失函数
loss_fn = nn.NLLLoss()
# 优化器: 用于自动优化每个追踪了梯度的参数
# 优化函数不止 w = w - lr * w.grad，优化器提供了各式各样的优化方法
# model.parameters(): 返回模型中用于训练的参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    # 清空梯度
    optimizer.zero_grad()
    # 前向传播
    _y = model(x)
    # 求损失
    loss = loss_fn(_y, y)
    print(loss.item())
    # 反向传播
    loss.backward()
    # 优化参数
    optimizer.step()

idx = random.randint(0, 63)

_y = model(x[idx].unsqueeze(0))
print(_y)
print(torch.argmax(_y, dim=-1))
print(y[idx])
