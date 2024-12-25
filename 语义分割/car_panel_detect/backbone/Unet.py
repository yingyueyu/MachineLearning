import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self, num_classes,device):
        super(Unet, self).__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = device
            raise IOError("设备不可以为空")

        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)

        self.conv5 = self.conv_block(512, 1024)

        self.conv6 = self.conv_block(1024, 512)
        self.conv7 = self.conv_block(512, 256)
        self.conv8 = self.conv_block(256, 128)
        self.conv9 = self.conv_block(128, 64)

        self.conv_f = nn.Conv2d(64, num_classes, 1, 1,device=self.device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1,device=self.device),
            nn.BatchNorm2d(out_channels,device=self.device),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,device=self.device),
            nn.BatchNorm2d(out_channels,device=self.device),
            nn.ReLU()
        )

    def upsample(self, conv_result, conv_skip):
        out_channels = conv_result.shape[1]
        layer = nn.ConvTranspose2d(out_channels, out_channels // 2, 2, 2,device=self.device)
        result = layer(conv_result)
        return torch.concatenate([conv_skip, result], dim=1)

    def forward(self, x):
        k1 = x = self.conv1(x)
        x = self.max_pool(x)
        k2 = x = self.conv2(x)
        x = self.max_pool(x)
        k3 = x = self.conv3(x)
        x = self.max_pool(x)
        k4 = x = self.conv4(x)
        x = self.max_pool(x)

        x = self.conv5(x)

        x = self.upsample(x, k4)
        x = self.conv6(x)
        x = self.upsample(x, k3)
        x = self.conv7(x)
        x = self.upsample(x, k2)
        x = self.conv8(x)
        x = self.upsample(x, k1)
        x = self.conv9(x)

        return self.softmax(self.conv_f(x))


# if __name__ == '__main__':
#     model = Unet(2)
#     images = torch.randn(1, 3, 448, 448)
#     result = model(images)
#     print(result.shape)
