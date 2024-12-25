import torch
from torch import nn
from torchvision.ops import MLP, Conv2dNormActivation

# Conv2dNormActivation: 这个模块包含了 卷积 + 归一化 + 激活

# BasicBlock 基础残差块
class BasicBlock(nn.Module):
    # down_sampling: 是否下采样
    def __init__(self, input_channels, output_channels, down_sampling=False):
        super().__init__()
        self.conv1 = Conv2dNormActivation(input_channels, output_channels, 3, padding=1,
                                          stride=2 if down_sampling else 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2 = Conv2dNormActivation(output_channels, output_channels, 3, padding=1, stride=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=None)
        # 残差连接下采样时的卷积
        self.res_conv = Conv2dNormActivation(input_channels, output_channels, 1,
                                             stride=2 if down_sampling else 1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=None) \
            if down_sampling or input_channels != output_channels else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 恒等映射
        identity = x
        # 卷积
        x = self.conv1(x)
        x = self.conv2(x)
        # 判断是否需要下采样
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        # 残差连接
        x += identity
        y = self.relu(x)
        return y


# Bottlenect 瓶颈残差块
class Bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, down_sampling=False):
        super().__init__()
        # 卷积中间层的输入输出应该是输出通道的四分之一
        d_4 = int(output_channels * 0.25)
        self.conv1 = Conv2dNormActivation(input_channels, d_4, 1, stride=2 if down_sampling else 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2 = Conv2dNormActivation(d_4, d_4, 3, padding=1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv3 = Conv2dNormActivation(d_4, output_channels, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=None)

        # 残差连接下采样时的卷积
        self.res_conv = Conv2dNormActivation(input_channels, output_channels, 1, stride=2 if down_sampling else 1,
                                             norm_layer=nn.BatchNorm2d,
                                             activation_layer=None) \
            if down_sampling or input_channels != output_channels else None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 恒等映射
        identity = x

        # 卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 判断是否下采样
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        x += identity
        y = self.relu(x)
        return y


# 创建ResNet中的循环卷积层
# block: 残差块的类型
# input_channels: 输入通道数
# output_channels: 输出通道数
# num_blocks: 残差块的数量
# down_sampling: 是否下采样
def make_layer(block, input_channels, output_channels, num_blocks, down_sampling=False):
    return nn.Sequential(*[block(input_channels, output_channels, down_sampling) if i == 0
                           else block(output_channels, output_channels) for i in range(num_blocks)])


# ResNet34
class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 64, 7, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2_x = make_layer(BasicBlock, 64, 64, 3)
        self.conv3_x = make_layer(BasicBlock, 64, 128, 4, down_sampling=True)
        self.conv4_x = make_layer(BasicBlock, 128, 256, 6, down_sampling=True)
        self.conv5_x = make_layer(BasicBlock, 256, 512, 3, down_sampling=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool(x)
        # N x 64 x 56 x 56
        x = self.conv2_x(x)
        # N x 64 x 56 x 56
        x = self.conv3_x(x)
        # N x 128 x 28 x 28
        x = self.conv4_x(x)
        # N x 256 x 14 x 14
        x = self.conv5_x(x)
        # N x 512 x 7 x 7
        x = self.avg_pool(x)
        # N x 512 x 1 x 1
        x = self.flatten(x)
        # N x 512
        x = self.fc(x)
        # N x 1000
        y = self.log_softmax(x)
        return y


# ResNet50
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 64, 7, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2_x = make_layer(Bottleneck, 64, 256, 3, False)
        self.conv3_x = make_layer(Bottleneck, 256, 512, 4, True)
        self.conv4_x = make_layer(Bottleneck, 512, 1024, 6, True)
        self.conv5_x = make_layer(Bottleneck, 1024, 2048, 3, True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(2048, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool(x)
        # N x 64 x 56 x 56
        x = self.conv2_x(x)
        # N x 256 x 56 x 56
        x = self.conv3_x(x)
        # N x 512 x 28 x 28
        x = self.conv4_x(x)
        # N x 1024 x 14 x 14
        x = self.conv5_x(x)
        # N x 2048 x 7 x 7
        x = self.avg_pool(x)
        # N x 2048 x 1 x 1
        x = self.flatten(x)
        # N x 2048
        x = self.fc(x)
        # N x 1000
        y = self.log_softmax(x)
        return y


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.conv1 = Conv2dNormActivation(3, 64, 7, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)

        is_bottleneck = block == Bottleneck

        output_channels = 64 * 4 if is_bottleneck else 64

        self.conv2_x = self._make_layer(block, 64, output_channels, num_blocks[0])
        self.conv3_x = self._make_layer(block, output_channels, output_channels * 2, num_blocks[1], True)
        self.conv4_x = self._make_layer(block, output_channels * 2, output_channels * 2 ** 2, num_blocks[2], True)
        self.conv5_x = self._make_layer(block, output_channels * 2 ** 2, output_channels * 2 ** 3, num_blocks[3], True)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(output_channels * 2 ** 3, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks, down_sampling=False):
        return nn.Sequential(*[block(input_channels, output_channels, down_sampling) if i == 0
                               else block(output_channels, output_channels) for i in range(num_blocks)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
    y = resnet18(x)
    print(y.shape)
    y = resnet101(x)
    print(y.shape)
