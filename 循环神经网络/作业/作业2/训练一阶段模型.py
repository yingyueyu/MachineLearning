import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from 数据集 import ImageDataset
from 模型 import ImageRecognition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 100
batch_size = 9
lr = 1e-2

# 数据集
ds = ImageDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = ImageRecognition()
model.to(device)

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

model.train()
for epoch in range(EPOCH):
    for i, (img, label) in enumerate(dl):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        y = model(img)
        loss = loss_fn(y, label)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}], loss: {total_loss / count}')

print(f'train over; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/R.pt')
