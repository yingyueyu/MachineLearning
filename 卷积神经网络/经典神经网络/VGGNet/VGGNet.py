import torch
from torch import nn
import re


class VGGNet16(nn.Module):
    # num_classes: 分类个数
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        # 卷积特征提取器
        # nn.Sequential 模块序列
        # 当调用该序列时，会依次调用内部的模块
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    # x: (N, 3, 224, 224)
    def forward(self, x):
        x = self.conv_layer1(x)
        # N x 64 x 112 x 112
        x = self.conv_layer2(x)
        # N x 128 x 56 x 56
        x = self.conv_layer3(x)
        # N x 256 x 28 x 28
        x = self.conv_layer4(x)
        # N x 512 x 14 x 14
        x = self.conv_layer5(x)
        # N x 512 x 7 x 7

        x = torch.flatten(x, start_dim=1)
        # N x 25088
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    model = VGGNet16()
    y = model(x)
    # print(y.shape)

    regex = '^conv_layer[1-5]'

    for name, module in model.named_modules():
        print(name)
        print(module)
        # 查看所有名字以 '^conv_layer[1-5]' 开头的模块
        if re.search(name, regex):
            # 判断这个模块是不是卷积
            if isinstance(module, nn.Conv2d):
                for p in module.parameters():
                    # 冻结参数，让该参数不要追踪梯度
                    p.requires_grad = False
