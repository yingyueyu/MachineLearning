# 数字识别
import torch
from torch import nn


class NumberRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)

        self.mp = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1620, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    # x: Nx1x28x28
    def forward(self, x):
        # Nx1x28x28
        x = self.conv1(x)
        x = self.bn1(x)
        # Nx10x26x26
        x = self.mp(x)
        # Nx10x13x13
        x = self.conv2(x)
        x = self.bn2(x)
        # Nx20x9x9
        x = self.flatten(x)
        # Nx1620

        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)

        return y


if __name__ == '__main__':
    model = NumberRecognition()
    x = torch.randn(5, 1, 28, 28)
    print(model(x).shape)
