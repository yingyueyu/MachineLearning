import torch
from torch import nn


class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def conv2dt_layer(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, stride, stride)
            )

        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv4t = conv2dt_layer(512, num_classes, 1)
        self.conv5 = conv_block(512, 512)
        self.conv6 = conv_block(512, 4096)
        self.conv7t = conv2dt_layer(4096, num_classes, 2)
        self.conv8t = conv2dt_layer(num_classes, num_classes, 16)

        self.dropout = nn.Dropout()
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x1 = self.conv4t(x)
        x = self.pool(self.conv5(x))
        x = self.relu(self.bn(self.conv6(x)))
        x2 = self.conv7t(x)
        x = x1 + x2
        x = self.conv8t(x)
        return x


# image = torch.randn((1, 3, 512, 512))
# fcn32s = FCN16s(2)
# result = fcn32s(image)
# print(result.shape)
