from torch import nn
import torch


class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        # return_indices=True 返回最大池化下标位置
        self.max_pool = nn.MaxPool2d(2, 2, return_indices=True)

        # 反最大池化
        self.max_un_pool = nn.MaxUnpool2d(2, 2)

        self.conv1 = self.conv_block(3, 64, 2)
        self.conv2 = self.conv_block(64, 128, 2)
        self.conv3 = self.conv_block(128, 256, 3)
        self.conv4 = self.conv_block(256, 512, 3)
        self.conv5 = self.conv_block(512, 1024, 3)

        self.conv6 = self.conv_block(1024, 512, 3)
        self.conv7 = self.conv_block(512, 256, 3)
        self.conv8 = self.conv_block(256, 128, 3)
        self.conv9 = self.conv_block(128, 64, 2)
        self.conv10 = self.conv_block(64, 64, 2)

        self.conv_f = nn.Conv2d(64, num_classes, 1, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def conv_block(self, in_channels, out_channels, layer_num):
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x, i1 = self.max_pool(self.conv1(x))
        x, i2 = self.max_pool(self.conv2(x))
        x, i3 = self.max_pool(self.conv3(x))
        x, i4 = self.max_pool(self.conv4(x))
        x, i5 = self.max_pool(self.conv5(x))

        x = self.conv6(self.max_un_pool(x, i5))
        x = self.conv7(self.max_un_pool(x, i4))
        x = self.conv8(self.max_un_pool(x, i3))
        x = self.conv9(self.max_un_pool(x, i2))
        x = self.conv10(self.max_un_pool(x, i1))

        return self.softmax(self.conv_f(x))


if __name__ == '__main__':
    model = SegNet(2)
    images = torch.randn(1, 3, 448, 448)
    result = model(images)
    print(result.shape)
