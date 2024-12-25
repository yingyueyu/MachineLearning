import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


# 深度可分离卷积
class DSConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.depth_wise = Conv2dNormActivation(input_channels, input_channels, 3, padding=1, groups=input_channels,
                                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.point_wise = Conv2dNormActivation(input_channels, output_channels, 1,
                                               norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, x):
        x = self.depth_wise(x)
        y = self.point_wise(x)
        return y


# 倒置残差块
class Bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, t=1, down_sampling=False):
        super().__init__()
        # 第一个1x1卷积的输出通道数
        output_conv1x1 = input_channels * t
        self.conv1x1 = Conv2dNormActivation(input_channels, output_conv1x1, 1,
                                            stride=2 if down_sampling else 1,
                                            norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.dsconv = DSConv(output_conv1x1, output_channels)
        # 是否做残差连接
        # 当不做下采样且输入输出通道相同时 才做残差连接
        self.is_res = not down_sampling and input_channels == output_channels

    def forward(self, x):
        identity = None
        # 恒等映射
        if self.is_res:
            identity = x

        # 第一个 1x1 卷积 用于减少参数量
        x = self.conv1x1(x)
        # 深度可分离卷积
        x = self.dsconv(x)
        # 深度可分离卷积最后的逐点卷积采用线性瓶颈，此处不用激活
        # 所以残差后也不用激活，否则时去了线性瓶颈的意义
        if self.is_res:
            x += identity
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 32, 3, padding=1, stride=2,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.bottleneck2 = self._make_layer(32, 16, t=1, n=1)
        self.bottleneck3 = self._make_layer(16, 24, t=6, n=2, down_sampling=True)
        self.bottleneck4 = self._make_layer(24, 32, t=6, n=3, down_sampling=True)
        self.bottleneck5 = self._make_layer(32, 64, t=6, n=4, down_sampling=True)
        self.bottleneck6 = self._make_layer(64, 96, t=6, n=3)
        self.bottleneck7 = self._make_layer(96, 160, t=6, n=3, down_sampling=True)
        self.bottleneck8 = self._make_layer(160, 320, t=6, n=1)
        self.conv9 = Conv2dNormActivation(320, 1280, 1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        self.avg_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(1280, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 32 x 112 x 112
        x = self.bottleneck2(x)
        # N x 16 x 112 x 112
        x = self.bottleneck3(x)
        # N x 24 x 56 x 56
        x = self.bottleneck4(x)
        # N x 32 x 28 x 28
        x = self.bottleneck5(x)
        # N x 64 x 14 x 14
        x = self.bottleneck6(x)
        # N x 96 x 14 x 14
        x = self.bottleneck7(x)
        # N x 160 x 7 x 7
        x = self.bottleneck8(x)
        # N x 320 x 7 x 7
        x = self.conv9(x)
        # N x 1280 x 7 x 7
        x = self.avg_pool(x)
        # N x 1280 x 1 x 1
        x = self.flatten(x)
        # N x 1280
        x = self.fc(x)
        # N x 1000
        y = self.log_softmax(x)
        return y

    # n: bottleneck重复的次数
    def _make_layer(self, input_channels, output_channels, t, n, down_sampling=False):
        # 第一次循环需要进行下采样，而其他次数的循环不需要下采样
        # 第二次循环开始输入通道数就应该等于 output_channels
        return nn.Sequential(*[Bottleneck(input_channels, output_channels, t=t, down_sampling=down_sampling)
                               if i == 0 else Bottleneck(output_channels, output_channels, t=t) for i in range(n)])


if __name__ == '__main__':
    model = MobileNetV2()
    x = torch.randn(5, 3, 224, 224)
    y = model(x)
    print(y.shape)
    # 统计参数数量
    for n, p in model.named_parameters():
        print(n)
        print(p.numel())
    print(sum([p.numel() for p in model.parameters()]))
