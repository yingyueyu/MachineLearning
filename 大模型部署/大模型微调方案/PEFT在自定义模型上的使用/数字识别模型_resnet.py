import torch
import torchvision
from torch import nn


class NumberRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用空模型，不下载预训练参数
        self.resnet = torchvision.models.resnet34()
        # 修改第一层输入，但是模型要求的输入图片依然是 224x224
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改最后层输出，输出个数为 10 代表 10 个数字的分类类数
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    model = NumberRecognition()
    x = torch.randn(5, 1, 224, 224)
    y = model(x)
    print(y.shape)
