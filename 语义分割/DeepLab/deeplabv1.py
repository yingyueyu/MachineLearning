import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights


class DeepLabV1(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV1, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 加载全连接层FC之前的所有的卷积和池化层
        self.feature_model = vgg_model.features
        # 切割原来的模型
        # cut_model = nn.Sequential(*list(vgg_model.features)[:10])
        # 修改前三个池化层为 3,2,1
        self.feature_model[4] = nn.MaxPool2d(3, 2, 1)
        self.feature_model[9] = nn.MaxPool2d(3, 2, 1)
        self.feature_model[16] = nn.MaxPool2d(3, 2, 1)
        # 后两个池化层 为 3,1,1
        self.feature_model[23] = nn.MaxPool2d(3, 1, 1)
        self.feature_model[30] = nn.MaxPool2d(3, 1, 1)
        # 改造 最后三个卷积为空洞卷积
        self.feature_model[24] = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.feature_model[26] = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.feature_model[28] = nn.Conv2d(512, 512, 3, 1, 2, 2)

        self.conv_f = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 12, 12),
            nn.Dropout(),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.Dropout(),
            nn.Conv2d(1024, num_classes, 1, 1),
            nn.Upsample(scale_factor=8)
        )

    def forward(self, x):
        x = self.feature_model(x)
        x = self.conv_f(x)
        return x


if __name__ == '__main__':
    model = DeepLabV1(2)
    images = torch.randn(1, 3, 224, 224)
    result = model(images)
    print(result.shape)
