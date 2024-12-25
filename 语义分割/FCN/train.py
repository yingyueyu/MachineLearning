import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from FCN32s import FCN32s


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = CustomDataset(torch.load("car_train_objection.pth", weights_only=False))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_dataset = CustomDataset(torch.load("car_valid_objection.pth", weights_only=False))
valid_loader = DataLoader(train_dataset, batch_size=10)

model = FCN32s(2)

# 定义训练设备
device = torch.device("cuda:0")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 1
# 最佳最准确率
best_mIOU = -1
for epoch in range(num_epochs):
    total_loss = 0.0
    mIOU = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images.float())
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    with torch.no_grad():
        for images, labels in valid_loader:
            val_images = images.to(device)
            val_labels = labels.to(device)

            val_outputs = model(val_images.float())
            _, predicted = val_outputs.max(1)
            total += val_labels.size(0)
            TP = (predicted & val_labels).sum().item()
            F = (predicted != val_labels).sum().item()
            mIOU += TP / (TP + F)

    # 打印训练信息
    print('Epoch [{}/{}], Loss: {:.4f}, mIoU: {:.2f}%'
          .format(epoch + 1, num_epochs, total_loss / len(train_loader), 100 * mIOU / total))

    if 100 * mIOU / total > best_mIOU:
        best_mIOU = 100 * mIOU / total
        torch.save(model.state_dict(), 'best.pth')
