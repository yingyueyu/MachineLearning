# 并行运算 Injection 模块
import torch
from torch import nn


class Injection(nn.Module):
    def __init__(self, input_channels, c1x1, rdc3x3, c3x3, rdc5x5, c5x5, pool_proj):
        super().__init__()
        # injection 的 4 分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, c1x1, 1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, rdc3x3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdc3x3, c3x3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, rdc5x5, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rdc5x5, c5x5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, 1),
            nn.ReLU(inplace=True),
        )

    # x (N, input_size, H, W)
    def forward(self, x):
        # 并行运算
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        # 拼接
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return y


# 辅助分类器
class InjectionAux(nn.Module):
    def __init__(self, input_channels, num_classes=1000, dropout=0.7):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=3)
        self.conv = nn.Conv2d(input_channels, 128, 1)
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.injection3a = Injection(192, 64, 96, 128, 16, 32, 32)
        self.injection3b = Injection(256, 128, 128, 192, 32, 96, 64)
        self.injection4a = Injection(480, 192, 96, 208, 16, 48, 64)
        self.injection4b = Injection(512, 160, 112, 224, 24, 64, 64)
        self.injection4c = Injection(512, 128, 128, 256, 24, 64, 64)
        self.injection4d = Injection(512, 112, 144, 288, 32, 64, 64)
        self.injection4e = Injection(528, 256, 160, 320, 32, 128, 128)
        self.injection5a = Injection(832, 256, 160, 320, 32, 128, 128)
        self.injection5b = Injection(832, 384, 192, 384, 48, 128, 128)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = InjectionAux(512, num_classes)
        self.aux2 = InjectionAux(528, num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 192 x 56 x 56
        x = self.max_pool(x)
        # N x 192 x 28 x 28
        x = self.injection3a(x)
        # N x 256 x 28 x 28
        x = self.injection3b(x)
        # N x 480 x 28 x 28
        x = self.max_pool(x)
        # N x 480 x 14 x 14
        x = self.injection4a(x)
        # N x 512 x 14 x 14

        # 辅助分类器
        # 仅当训练时使用辅助分类器
        y1 = self.aux1(x) if self.training else None

        x = self.injection4b(x)
        # N x 512 x 14 x 14
        x = self.injection4c(x)
        # N x 512 x 14 x 14
        x = self.injection4d(x)
        # N x 528 x 14 x 14

        # 仅当训练时使用辅助分类器
        y2 = self.aux2(x) if self.training else None

        x = self.injection4e(x)
        # N x 832 x 14 x 14
        x = self.max_pool(x)
        # N x 832 x 7 x 7
        x = self.injection5a(x)
        # N x 832 x 7 x 7
        x = self.injection5b(x)
        # N x 1024 x 7 x 7
        # 全局平均池化
        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = self.dropout(x)

        x = self.flatten(x)
        # N x 1024
        y = self.fc(x)
        # N x 1000 (num_classes)
        return y, y1, y2


if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    label = torch.tensor([0, 1, 2, 3, 4])
    model = GoogLeNet()
    model.train()
    y, y1, y2 = model(x)
    print(y.shape)
    print(y1)
    print(y2)

    # 计算损失
    loss_fn = nn.CrossEntropyLoss()
    loss1 = loss_fn(y, label)
    loss2 = loss_fn(y1, label) * 0.3
    loss3 = loss_fn(y2, label) * 0.3
    loss = loss1 + loss2 + loss3
    print(loss)
    loss.backward()
    print('反向传播完成')

    model.eval()
    y, y1, y2 = model(x)
    print(y.shape)
    print(y1)
    print(y2)

