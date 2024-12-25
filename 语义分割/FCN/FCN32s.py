import torch
from torch import nn


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def con_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.block1 = con_block(3, 64)
        self.block2 = con_block(64, 128)
        self.block3 = con_block(128, 256)
        self.block4 = con_block(256, 512)
        self.block5 = con_block(512, 512)
        self.block6 = con_block(512, 4096)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(4096)
        self.softmax = nn.Softmax(1)

        # 转置卷积
        self.conv_t = nn.ConvTranspose2d(4096, num_classes + 1, 32, 32)

    def forward(self, x):
        x = self.max_pool(self.block1(x))
        x = self.max_pool(self.block2(x))
        x = self.max_pool(self.block3(x))
        x = self.max_pool(self.block4(x))
        x = self.max_pool(self.block5(x))
        x = self.relu(self.bn(self.block6(x)))

        return self.softmax(self.conv_t(x))


if __name__ == '__main__':
    model = FCN32s(1)
    image = torch.randn(1, 3, 512, 512)
    result = model(image)
    print(result.shape)
