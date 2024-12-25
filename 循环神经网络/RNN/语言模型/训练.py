import torch
from torch import nn
from torch.utils.data import DataLoader

from 数据集与分词器 import WordDataset
from 语言模型 import LangModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 10000
batch_size = 8
lr = 0.01

# 数据集
ds = WordDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = LangModel()
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
        y, h = model(inputs)
        # 对 y 和 label 变形，让交叉熵能够识别
        y = y.reshape(y.shape[0] * y.shape[1], 16)
        labels = labels.reshape(labels.numel())
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
