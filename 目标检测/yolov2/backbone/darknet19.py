import torch
from torch import nn

"""
YOLOv2 的backbone --- darknet-19
"""


class DarkNet19(nn.Module):
    def __init__(self, anchors_num=5, classes_num=20):
        super().__init__()

        def conv_block(in_channels, out_channels, num_conv=1):
            layers = []
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels)
            ))
            if num_conv > 1 and num_conv % 2 == 1:
                for _ in range(num_conv - 1):
                    layers.append(nn.Sequential(
                        nn.Conv2d(out_channels, in_channels, 1, 1),
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                        nn.BatchNorm2d(out_channels)
                    ))
            return nn.Sequential(*layers)

        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv_1 = conv_block(3, 32)
        self.conv_2 = conv_block(32, 64)
        self.conv_3 = conv_block(64, 128, 3)
        self.conv_4 = conv_block(128, 256, 3)
        self.conv_5 = conv_block(256, 512, 5)
        self.conv_6 = conv_block(512, 1024, 5)

        self.conv_f = nn.ConvTranspose2d(1024, 1000, 1, 1)
        self.avg_pool = nn.AvgPool2d(7, 7)
        self.softmax = nn.Softmax(dim=-1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(1000, 500, 1, 1),
            nn.Conv2d(500, 250, 1, 1),
            nn.Conv2d(250, 125, 1, 1)
        )

    def forward(self, x):
        x = self.max_pool(self.conv_1(x))
        x = self.max_pool(self.conv_2(x))
        x = self.max_pool(self.conv_3(x))
        x = self.max_pool(self.conv_4(x))
        x = self.max_pool(self.conv_5(x))
        x = self.conv_6(x)
        x = self.conv_f(x)
        x = self.avg_pool(x)
        return self.conv3(x).view(-1)


if __name__ == '__main__':
    image = torch.randn(1, 3, 416, 416)
    net = DarkNet19()
    result = net(image)
    print(result.shape)
