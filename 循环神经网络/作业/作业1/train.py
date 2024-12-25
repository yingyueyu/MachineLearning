import torch
from torch import nn
from torch.utils.data import DataLoader

from Dataset import TempDataset
from TempModel import TempModel

torch.manual_seed(100)

# 超参数
EPOCH = 30000
batch_size = 15
lr = 1e-2

# 数据集
ds = TempDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = TempModel(10)

# 损失函数 优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

# 训练
model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(dl):
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        y, h = model(inputs.unsqueeze(2))
        # 计算损失
        loss = loss_fn(y[:, -1].squeeze(1), labels)
        total_loss += loss.item()
        count += 1
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

print(f'训练完成 loss: {total_loss / count}')
torch.save(model.state_dict(), 'model.pt')
