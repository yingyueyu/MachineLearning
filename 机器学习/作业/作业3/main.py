import torch
from torch import nn

lr = 1e-2

# torch.manual_seed(100)

x = torch.randint(0, 3, (100, 3)).to(torch.float)
print(x)
print(x.shape)
y = torch.randint(0, 2, (100, 1)).to(torch.float)
print(y)
print(y.shape)

fc1 = nn.Linear(3, 10)
fc2 = nn.Linear(10, 1)


def model(x):
    x = fc1(x)
    x = torch.nn.functional.relu(x)
    y = fc2(x)
    # y = torch.nn.functional.sigmoid(y)
    return y


# 损失函数
# loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(1000):
    # 2. 前向传播
    _y = model(x)
    # 3. 计算损失
    loss = loss_fn(_y, y)
    print(loss.item())
    # 4. 反向传播
    loss.backward()
    # 5. 更新参数
    # fc1.weight 是 nn.Parameter 对象，这是一个会被追踪梯度的对象
    # fc1.weight.data 是 nn.Parameter 对象的 data 属性，这是一个不会被追踪梯度的用于存储张量数据的对象
    fc1.weight.data -= lr * fc1.weight.grad
    fc1.bias.data -= lr * fc1.bias.grad
    fc2.weight.data -= lr * fc2.weight.grad
    fc2.bias.data -= lr * fc2.bias.grad
    # 1. 清空梯度
    fc1.weight.grad.zero_()
    fc1.bias.grad.zero_()
    fc2.weight.grad.zero_()
    fc2.bias.grad.zero_()

# 随机索引
idx = torch.randint(0, 100, (1,))[0].item()

_y = model(x[idx])
# 激活并得到事件发生的概率
_y = nn.functional.sigmoid(_y)
print(_y)
