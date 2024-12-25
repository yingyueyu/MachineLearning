import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        # 池化层
        self.pool = nn.AvgPool2d(2)
        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        # 激活函数
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=1)
        # 展平
        self.flatten = nn.Flatten(start_dim=1)

    # x 输入形状 (N, 1, 32, 32)
    def forward(self, x):
        x = self.conv1(x)
        # N x 6 x 28 x 28
        x = self.tanh(x)
        x = self.pool(x)
        # N x 6 x 14 x 14
        x = self.conv2(x)
        # N x 16 x 10 x 10
        x = self.tanh(x)
        x = self.pool(x)
        # N x 16 x 5 x 5
        x = self.conv3(x)
        # N x 120 x 1 x 1
        x = self.tanh(x)
        x = self.flatten(x)
        # N x 120
        x = self.fc1(x)
        # N x 84
        x = self.tanh(x)
        x = self.fc2(x)
        # N x 10
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    model = LeNet5()
    x = torch.randn(5, 1, 32, 32)
    y = model(x)
    print(y.shape)
