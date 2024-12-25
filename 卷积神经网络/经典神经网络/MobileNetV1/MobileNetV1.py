import math

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import Resize


# 深度可分离卷积 Depthwise Separable Convolution
class DSConv(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        # 深度卷积
        self.depthwise = Conv2dNormActivation(input_channels, input_channels, 3, padding=1,
                                              stride=stride, groups=input_channels,
                                              norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        # 逐点卷积
        self.pointwise = Conv2dNormActivation(input_channels, output_channels, 1,
                                              norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    # alpha: 宽度系数，除了第一个卷积层的输入以外，其余卷积层的通道数都乘以这个 alpha 系数
    # rho: 分辨率系数，将输入图片宽高重置大小为 rho * 224
    def __init__(self, num_classes=1000, alpha=1., rho=1.):
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self._get_rho_transform()
        self.conv1 = Conv2dNormActivation(3, self._alpha_channel_num(32), 3, padding=1, stride=2,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv2 = DSConv(self._alpha_channel_num(32), self._alpha_channel_num(64))
        self.conv3 = DSConv(self._alpha_channel_num(64), self._alpha_channel_num(128), stride=2)
        self.conv4 = DSConv(self._alpha_channel_num(128), self._alpha_channel_num(128))
        self.conv5 = DSConv(self._alpha_channel_num(128), self._alpha_channel_num(256), stride=2)
        self.conv6 = DSConv(self._alpha_channel_num(256), self._alpha_channel_num(256))
        self.conv7 = DSConv(self._alpha_channel_num(256), self._alpha_channel_num(512), stride=2)
        self.conv8 = nn.Sequential(
            *[DSConv(self._alpha_channel_num(512), self._alpha_channel_num(512)) for i in range(5)])
        self.conv9 = DSConv(self._alpha_channel_num(512), self._alpha_channel_num(1024), stride=2)
        # 通过卷积后图片大小计算公式，套用公式计算对应的 padding
        padding = int(math.ceil((self.pool_kernel_size + 1) * 0.5))
        self.conv10dw = Conv2dNormActivation(self._alpha_channel_num(1024), self._alpha_channel_num(1024), 3, stride=2,
                                             groups=self._alpha_channel_num(1024), padding=padding,
                                             norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.conv10pw = Conv2dNormActivation(self._alpha_channel_num(1024), self._alpha_channel_num(1024), 1,
                                             norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.avg_pool = nn.AvgPool2d(self.pool_kernel_size)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self._alpha_channel_num(1024), num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _alpha_channel_num(self, channel):
        return int(self.alpha * channel)

    # 构造 rho 转换器
    def _get_rho_transform(self):
        size = int(self.rho * 224)
        # MNV1 模型的卷积部分将原图缩小 32 倍
        # 所以通过 rho 算出的新图片，需要被 32 整除
        rest = size % 32
        if rest != 0:
            # 让新图片大小添加一个 32 的补数
            size += 32 - rest
        # 计算全局平均池化的池化核大小
        self.pool_kernel_size = size // 32
        self.rho_transform = Resize((size, size), antialias=True)

    def forward(self, x):
        # 通过系数 rho 降低图片大小
        x = self.rho_transform(x)
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 32 x 112 x 112
        x = self.conv2(x)
        # N x 64 x 112 x 112
        x = self.conv3(x)
        # N x 128 x 56 x 56
        x = self.conv4(x)
        # N x 128 x 56 x 56
        x = self.conv5(x)
        # N x 256 x 28 x 28
        x = self.conv6(x)
        # N x 256 x 28 x 28
        x = self.conv7(x)
        # N x 512 x 14 x 14
        x = self.conv8(x)
        # N x 512 x 14 x 14
        x = self.conv9(x)
        # N x 1024 x 7 x 7
        x = self.conv10dw(x)
        # N x 1024 x 7 x 7
        x = self.conv10pw(x)
        # N x 1024 x 7 x 7
        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = self.flatten(x)
        # N x 1024
        x = self.fc(x)
        # N x 1000
        y = self.log_softmax(x)
        return y


# class MobileNetV1(nn.Module):
#     def __init__(self, num_classes=1000):
#         super().__init__()
#         self.conv1 = Conv2dNormActivation(3, 32, 3, padding=1, stride=2,
#                                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
#         self.conv2 = DSConv(32, 64)
#         self.conv3 = DSConv(64, 128, stride=2)
#         self.conv4 = DSConv(128, 128)
#         self.conv5 = DSConv(128, 256, stride=2)
#         self.conv6 = DSConv(256, 256)
#         self.conv7 = DSConv(256, 512, stride=2)
#         self.conv8 = nn.Sequential(*[DSConv(512, 512) for i in range(5)])
#         self.conv9 = DSConv(512, 1024, stride=2)
#         self.conv10dw = Conv2dNormActivation(1024, 1024, 3, stride=2, groups=1024, padding=4,
#                                              norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
#         self.conv10pw = Conv2dNormActivation(1024, 1024, 1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
#         self.avg_pool = nn.AvgPool2d(7)
#         self.flatten = nn.Flatten(start_dim=1)
#         self.fc = nn.Linear(1024, num_classes)
#         self.log_softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.conv1(x)
#         # N x 32 x 112 x 112
#         x = self.conv2(x)
#         # N x 64 x 112 x 112
#         x = self.conv3(x)
#         # N x 128 x 56 x 56
#         x = self.conv4(x)
#         # N x 128 x 56 x 56
#         x = self.conv5(x)
#         # N x 256 x 28 x 28
#         x = self.conv6(x)
#         # N x 256 x 28 x 28
#         x = self.conv7(x)
#         # N x 512 x 14 x 14
#         x = self.conv8(x)
#         # N x 512 x 14 x 14
#         x = self.conv9(x)
#         # N x 1024 x 7 x 7
#         x = self.conv10dw(x)
#         # N x 1024 x 7 x 7
#         x = self.conv10pw(x)
#         # N x 1024 x 7 x 7
#         x = self.avg_pool(x)
#         # N x 1024 x 1 x 1
#         x = self.flatten(x)
#         # N x 1024
#         x = self.fc(x)
#         # N x 1000
#         y = self.log_softmax(x)
#         return y


if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    model = MobileNetV1()
    model2 = MobileNetV1(alpha=0.5, rho=0.57)
    # y = model(x)
    # print(y.shape)

    print(sum([p.numel() for p in model.parameters()]))
    print(sum([p.numel() for p in model2.parameters()]))
