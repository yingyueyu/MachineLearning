import torch
from torch import nn


# 封装一个包含了卷积 + 批量归一化 + 激活函数的基础卷积模块
class BasicConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.activation(x)
        return y


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv(3, 32, 3)
        self.conv2 = BasicConv(32, 64, 3)
        self.conv3 = BasicConv(64, 128, 3)
        self.conv4 = BasicConv(128, 64, 1, padding=0)
        self.conv5 = BasicConv(64, 128, 3)
        self.conv6 = BasicConv(128, 256, 3)
        self.conv7 = BasicConv(256, 128, 1, padding=0)
        self.conv8 = BasicConv(128, 256, 3)
        self.conv9 = BasicConv(256, 512, 3)
        self.conv10 = BasicConv(512, 256, 1, padding=0)
        self.conv11 = BasicConv(256, 512, 3)
        self.conv12 = BasicConv(512, 1024, 3)
        self.conv13 = BasicConv(1024, 512, 1, padding=0)
        self.conv14 = BasicConv(512, 1024, 3)

        self.max_pool = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(7)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(1024, 1000)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 32 x 224 x 224
        x = self.max_pool(x)
        # N x 32 x 112 x 112
        x = self.conv2(x)
        # N x 64 x 112 x 112
        x = self.max_pool(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 128 x 56 x 56
        x = self.conv4(x)
        # N x 64 x 56 x 56
        x = self.conv5(x)
        # N x 128 x 56 x 56
        x = self.max_pool(x)
        # N x 128 x 28 x 28
        x = self.conv6(x)
        # N x 256 x 28 x 28
        x = self.conv7(x)
        # N x 128 x 28 x 28
        x = self.conv8(x)
        # N x 256 x 28 x 28
        x = self.max_pool(x)
        # N x 256 x 14 x 14
        x = self.conv9(x)
        # N x 512 x 14 x 14
        x = self.conv10(x)
        # N x 256 x 14 x 14
        x = self.conv11(x)
        # N x 512 x 14 x 14
        x = self.max_pool(x)
        # N x 512 x 7 x 7
        x = self.conv12(x)
        # N x 1024 x 7 x 7
        x = self.conv13(x)
        # N x 512 x 7 x 7
        x = self.conv14(x)
        # N x 1024 x 7 x 7
        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = self.flatten(x)
        # N x 1024
        x = self.fc(x)
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    model = Darknet19()
    x = torch.randn(5, 3, 224, 224)
    y = model(x)
    print(y.shape)
