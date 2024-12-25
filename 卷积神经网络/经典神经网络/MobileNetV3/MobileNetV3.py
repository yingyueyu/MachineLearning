import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


# 序列激发模块
class SEModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # 自适应平均池化，可以手动指定输出图像的大小
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        # 全连接层
        self.fc1 = nn.Linear(input_channels, input_channels // 4)
        self.fc2 = nn.Linear(input_channels // 4, input_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.hard_sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        y = self.hard_sigmoid(x)
        return y


# 倒置残差模块
class Bottleneck(nn.Module):
    # exp_size: 扩展卷积的输出通道数
    # output_channels: 输出通道数
    # SE: 是否使用SE模块
    # NL: 激活函数
    # s: 步长
    def __init__(self, input_channels, output_channels, exp_size, SE=False, NL=nn.ReLU, s=1):
        super().__init__()
        # 扩展卷积
        self.exp_conv = Conv2dNormActivation(input_channels, exp_size, 1, stride=s,
                                             norm_layer=nn.BatchNorm2d, activation_layer=NL)
        # 深度卷积
        self.depth_conv = Conv2dNormActivation(exp_size, exp_size, 3, padding=1, groups=exp_size,
                                               norm_layer=nn.BatchNorm2d, activation_layer=NL)
        # 创建序列激发模块
        self.se = SEModule(exp_size) if SE else None
        # 逐点卷积
        self.point_conv = Conv2dNormActivation(exp_size, output_channels, 1, stride=1,
                                               norm_layer=nn.BatchNorm2d, activation_layer=None)
        # 是否进行残差连接
        self.is_res = s == 1 and input_channels == output_channels

    def forward(self, x):
        # 判断是否要残差连接
        identity = x if self.is_res else None
        x = self.exp_conv(x)
        x = self.depth_conv(x)
        # 判断是否要序列激发
        if self.se is not None:
            weights = self.se(x)
            # 修改权重的形状
            weights = weights.view(x.shape[0], x.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            # x *= weights
            x = x * weights
        x = self.point_conv(x)
        # 残差连接
        if identity is not None:
            x += identity
        return x


# 序列激发模块
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 16, 3, stride=2, padding=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish)
        self.bneck2 = Bottleneck(16, 16, 16, SE=True, NL=nn.ReLU, s=2)
        self.bneck3 = Bottleneck(16, 24, 72, SE=False, NL=nn.ReLU, s=2)
        self.bneck4 = Bottleneck(24, 24, 88, SE=False, NL=nn.ReLU, s=1)
        self.bneck5 = Bottleneck(24, 40, 96, SE=True, NL=nn.Hardswish, s=2)
        self.bneck6 = Bottleneck(40, 40, 240, SE=True, NL=nn.Hardswish, s=1)
        self.bneck7 = Bottleneck(40, 40, 240, SE=True, NL=nn.Hardswish, s=1)
        self.bneck8 = Bottleneck(40, 48, 120, SE=True, NL=nn.Hardswish, s=1)
        self.bneck9 = Bottleneck(48, 48, 144, SE=True, NL=nn.Hardswish, s=1)
        self.bneck10 = Bottleneck(48, 96, 288, SE=True, NL=nn.Hardswish, s=2)
        self.bneck11 = Bottleneck(96, 96, 576, SE=True, NL=nn.Hardswish, s=1)
        self.bneck12 = Bottleneck(96, 96, 576, SE=True, NL=nn.Hardswish, s=1)
        self.conv13 = Conv2dNormActivation(96, 576, 1, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish)
        self.se13 = SEModule(576)
        self.avg_pool14 = nn.AvgPool2d(7)
        self.conv15 = Conv2dNormActivation(576, 1024, 1, norm_layer=None, activation_layer=nn.Hardswish)
        self.conv16 = nn.Conv2d(1024, num_classes, 1)
        self.flatten = nn.Flatten(start_dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 16 x 112 x 112
        x = self.bneck2(x)
        # N x 16 x 56 x 56
        x = self.bneck3(x)
        # N x 24 x 28 x 28
        x = self.bneck4(x)
        # N x 24 x 28 x 28
        x = self.bneck5(x)
        # N x 40 x 14 x 14
        x = self.bneck6(x)
        # N x 40 x 14 x 14
        x = self.bneck7(x)
        # N x 40 x 14 x 14
        x = self.bneck8(x)
        # N x 48 x 14 x 14
        x = self.bneck9(x)
        # N x 48 x 14 x 14
        x = self.bneck10(x)
        # N x 96 x 7 x 7
        x = self.bneck11(x)
        # N x 96 x 7 x 7
        x = self.bneck12(x)
        # N x 96 x 7 x 7
        x = self.conv13(x)
        weights = self.se13(x)
        weights = weights.view(x.shape[0], x.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        # x *= weights
        x = x * weights
        # N x 576 x 7 x 7
        x = self.avg_pool14(x)
        # N x 576 x 1 x 1
        x = self.conv15(x)
        # N x 1024 x 1 x 1
        x = self.conv16(x)
        # N x 1000 x 1 x 1
        x = self.flatten(x)
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    model = MobileNetV3()
    x = torch.randn(5, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(sum([p.numel() for p in model.parameters()]))
    loss_fn = nn.NLLLoss()
    loss = loss_fn(y, torch.tensor([0, 1, 2, 3, 4]))
    loss.backward()
