import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LotteryDataset
from 循环神经网络.作业.作业3.models import LotteryModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 1000
batch_size = 15
lr = 0.01

# 数据集
ds = LotteryDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = LotteryModel()
model.to(device)

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(dl):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y, h, c = model(inputs)
        y = y.reshape(-1, 17)
        labels = labels.reshape(-1)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

print('train over')

torch.save(model.state_dict(), 'weights/model.pt')
