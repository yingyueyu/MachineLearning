import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from 循环神经网络.作业.作业4.dataset import CardDataset
from models import CardModel_GRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 10
batch_size = 128
lr = 0.01

# 数据集
ds = CardDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = CardModel_GRU()
model.to(device)

# 损失函数 优化器
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

# 训练循环
model.train()
for epoch in range(EPOCH):
    print(f'epoch: [{epoch + 1}/{EPOCH}]')
    for i, (inputs, labels) in enumerate(dl):
        inputs = inputs.transpose(0, 1)
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        y, h = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optim.step()
        # 每十个批次打印数据
        if (i + 1) % 10 == 0:
            print(f'loss: {total_loss / count}')

print('train over')

print(f'loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
