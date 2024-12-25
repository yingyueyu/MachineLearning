import torch
from torch import nn


# 自定义神经网络需要继承 torch.nn.Module


# 自定义卷积基础模块
class BaseConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, _x):
        _x = self.conv(_x)
        _x = self.bn(_x)
        _y = self.relu(_x)
        return _y


# 水果分类器神经网络
class FruitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # # 卷积层
        # self.conv1 = nn.Conv2d(3, 256, 3)
        # self.conv2 = nn.Conv2d(256, 512, 3)
        # # 批量归一化
        # # 作用: 保留维度上的方向信息，将整个高维向量，长度变为1，这样有利于数学运算
        # self.bn1 = nn.BatchNorm2d(256)
        # self.bn2 = nn.BatchNorm2d(512)
        # # 激活: 非线性激活函数 relu 可以用来预测非线性的结果
        # self.relu = nn.ReLU()

        self.conv1 = BaseConv(3, 64, 3)
        self.conv2 = BaseConv(64, 32, 1)
        self.conv3 = BaseConv(32, 64, 4)
        self.conv4 = BaseConv(64, 128, 3)

        # 池化层
        self.max_pool = nn.MaxPool2d(2)

        # 展平
        self.flatten = nn.Flatten()

        # 全连接
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 3)
        # 最终全连接输出结果一定是3个，这个输出结果根据分类的类数而定
        # self.fc4 = nn.Linear(128, 3)

        # 激活函数
        self.relu = nn.ReLU()

    # 前向传播方法
    # 通俗讲就是模型的预测方法
    # _x 形状为 (N, 3, 100, 100); N: 代e，在此处也代表有几张图片
    def forward(self, _x):
        # 提取图像特征
        # # N x 3 x 100 x 100
        # _x = self.conv1(_x)
        # # N x 256 x 98 x 98
        # _x = self.bn1(_x)
        # # 激活
        # _x = self.relu(_x)
        # # N x 256 x 98 x 98
        # _x = self.conv2(_x)
        # # N x 512 x 96 x 96
        # _x = self.bn2(_x)
        # _x = self.relu(_x)

        # N x 3 x 100 x 100
        _x = self.conv1(_x)
        # N x 256 x 98 x 98
        _x = self.max_pool(_x)
        # N x 256 x 49 x 49
        _x = self.conv2(_x)
        # N x 128 x 49 x 49
        _x = self.conv3(_x)
        # N x 256 x 46 x 46
        _x = self.max_pool(_x)
        # N x 256 x 23 x 23
        _x = self.conv2(_x)
        # N x 128 x 23 x 23
        _x = self.conv3(_x)
        # N x 256 x 20 x 20
        _x = self.max_pool(_x)
        # N x 256 x 10 x 10
        _x = self.conv4(_x)
        # N x 512 x 8 x 8

        # 在分类前，将特征图进行展平
        _x = self.flatten(_x)
        # N x 32768

        # 通过特征分类
        _x = self.fc1(_x)
        _x = self.relu(_x)
        _x = self.fc2(_x)
        _x = self.relu(_x)
        _x = self.fc3(_x)
        # _x = self.relu(_x)
        # _x = self.fc4(_x)
        _y = self.relu(_x)

        return _y


if __name__ == '__main__':
    model = FruitClassifier()

    # 构造假数据
    x = torch.rand(2, 3, 100, 100)

    # 调用模型，会自动调用模型内的 forward 方法
    y = model(x)

    # y 形状为 (N, 3)
    # N: 有多少张图片的预测结果
    # 3: 每张图片预测的三分类的概率，最后去最大值的索引，作为代表水果的数字
    print(y)
    print(y.shape)
