import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from 数据集 import TextDataset
from RNN import RNN

# 超参数
EPOCH = 10000
batch_size = 10
lr = 0.01

# 创建数据
text = 'hey how are you'
text = list(set(text))
text = sorted(text)
print(text)


def transform(inp):
    # 输入序列
    # [
    #     [0 0 1 0 0 0 0 0 0],
    #     [0 0 0 0 0 0 1 0 0],
    #     [0 0 0 0 1 0 0 0 0],
    #     [0 0 1 0 0 0 0 0 0],
    #     [0 0 1 0 0 0 0 0 0],
    # ]
    idx = [text.index(c) for c in inp]
    t = F.one_hot(torch.tensor(idx), len(text))
    t = t.to(torch.float)
    return t


def target_transform(label):
    return torch.tensor(text.index(label))


ds = TextDataset(transform=transform, target_transform=target_transform)

# 创建模型
model = RNN(9, 9)

# 创建损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 开启训练模式
model.train()

total_loss = 0.
count = 0

# 训练模型
for epoch in range(EPOCH):
    for i in range(len(ds)):
        inp, label = ds[i]
        # 清空梯度
        optimizer.zero_grad()
        # 预测
        y, h = model(inp)
        # 计算损失
        loss = loss_fn(y[-1], label)
        total_loss += loss.item()
        count += 1
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
    print(f'epoch: [{epoch + 1}/{EPOCH}], avg_loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
