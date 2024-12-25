import torch
from torch import nn

"""
YOLOv2 的better改进 --- Batch Normal
                   --- 用卷积替代 池化 以及 全连接网络
"""


class YOLOv2Net(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv2Net, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, 2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 2, 1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, 2, 1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024)
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

        # 最后的1x1卷积层，通过改变通道数，得到最后的结果
        self.conv_f = nn.Conv2d(1024, num_classes + 5 * 2, 1, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv_f(x)
        return x.permute([0, 2, 3, 1])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLOv2Net(2)
    model = model.to(device)
    image = torch.randn(1, 3, 416, 416)
    result = model(image.to(device))

    print(result.shape)
