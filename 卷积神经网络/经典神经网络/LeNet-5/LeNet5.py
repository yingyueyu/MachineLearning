import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1
        self.c1 = nn.Conv2d(1, 6, 5)
        # 平均池化层
        self.ap = nn.AvgPool2d(2)
        # 卷积层2
        self.c2 = nn.Conv2d(6, 16, 5)
        # 卷积层3
        self.c3 = nn.Conv2d(16, 120, 5)
        # 全连接1
        self.fc1 = nn.Linear(120, 84)
        # 全连接2
        self.fc2 = nn.Linear(84, 10)

        # 激活函数
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # 展平
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.c1(x)
        # 每层后要激活
        x = self.tanh(x)
        x = self.ap(x)
        x = self.c2(x)
        x = self.tanh(x)
        x = self.ap(x)
        x = self.c3(x)
        # 先展平
        x = self.flatten(x)
        # 全连接分类
        x = self.fc1(x)
        x = self.tanh(x)
        y = self.fc2(x)
        # 求分类概率分布
        # 由于交叉熵代价函数自带softmax，该步骤可以省略
        # y = self.softmax(x)
        return y
