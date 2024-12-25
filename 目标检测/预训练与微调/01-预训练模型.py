import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights

# 设置设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义预训练模型
pretrained_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
pretrained_model.to(device)
# 冻结预训练模型的参数
for param in pretrained_model.parameters():
    param.requires_grad = False
# 替换最后一层全连接层
num_classes = 10  # 分类任务的类别数
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
pretrained_model.fc.to(device)
# 加载训练数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)
# 训练模型
num_epochs = 10
# 最佳最准确率
best_acc = -1
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = pretrained_model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 打印训练信息
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, total_loss / len(train_loader), 100 * correct / total))

    if 100 * correct / total > best_acc:
        best_acc = 100 * correct / total
        torch.save(pretrained_model.state_dict(), 'best.pth')
        # 此处在RTX 2080Ti的11G显存下，训练了10个epochs，准确率为80%
