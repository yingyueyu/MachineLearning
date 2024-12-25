from torch import nn
import torch
from backbone.Unet import Unet
from torch.utils.data import Dataset, DataLoader
import pynvml


class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


total_dataset = CustomDataset(torch.load("./data/seg/seg_images.pth", weights_only=False))
train_dataset = total_dataset[:800]
valid_dataset = total_dataset[800:]
total_labels = CustomDataset(torch.load("./data/seg/seg_labels.pth", weights_only=False))
train_labels = total_labels[:800]
valid_labels = total_labels[800:]

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False)
train_labels_loader = DataLoader(train_labels, batch_size=3, shuffle=False)
valid_labels_loader = DataLoader(valid_labels, batch_size=3, shuffle=False)

device = torch.device("cuda:0")
model = Unet(2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 显存监控（假如20batch_size 显存未满，则向上调解)）
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 训练模型
num_epochs = 1000
# 最佳最准确率
best_acc = -1
for epoch in range(num_epochs):
    total_loss = 0.0
    mIOU = 0
    total = 0
    for images, labels in zip(train_loader, train_labels_loader):
        images = images.to(device)
        labels = (labels / 255.).long()
        print(labels)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images.to(device).float())
        loss = criterion(outputs, labels.to(device))

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    with torch.no_grad():
        for images, labels in zip(valid_loader, valid_labels_loader):
            val_images = images.to(device)
            val_labels = labels.to(device)

            val_outputs = model(val_images.to(device).float())
            _, predicted = val_outputs.max(1)
            total += val_labels.size(0)
            TP = (predicted & val_labels).sum().item()
            F = (predicted != val_labels).sum().item()
            mIOU += TP / (TP + F)

            # 打印训练信息
    print('Epoch [{}/{}], Loss: {:.4f}, mIoU: {:.2f}%'
          .format(epoch + 1, num_epochs, total_loss / len(total_loss), 100 * mIOU / total))

    if 100 * mIOU / total > best_mIOU:
        best_mIOU = 100 * mIOU / total
        torch.save(model.state_dict(), 'best.pth')

    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU in use {(meminfo.used / 1024 ** 3):.4f}/{meminfo.total / 1024 ** 3} G')
