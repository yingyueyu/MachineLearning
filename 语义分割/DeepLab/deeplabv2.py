import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights


class DeepLabV2(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        # --------------- vgg部分--------------
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.font_conv = nn.Sequential(*list(vgg.features.children())[:23])
        # ----------------bottleneck部分---------
        self.layer3 = Layer3()
        self.layer4 = Layer4()
        self.branch_1 = nn.ConvTranspose2d(2048, num_classes, 3, 1, 6, dilation=6)
        self.branch_2 = nn.ConvTranspose2d(2048, num_classes, 3, 1, 12, dilation=12)
        self.branch_3 = nn.ConvTranspose2d(2048, num_classes, 3, 1, 18, dilation=18)
        self.branch_4 = nn.ConvTranspose2d(2048, num_classes, 3, 1, 24, dilation=24)

        self.up_sample = nn.Upsample(scale_factor=8)

    def forward(self, x):
        x = self.font_conv(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        x = x1 + x2 + x3 + x4
        return self.up_sample(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, p, r, is_add=True):
        super().__init__()
        self.is_add = is_add
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, p, r),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        if self.is_add:
            return x1 + x2
        else:
            return x2


class Layer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck1 = Bottleneck(512, 1024, 2, 2)
        self.bottleneck2 = Bottleneck(1024, 1024, 2, 2, False)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck2(x)
        return x


class Layer4(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck1 = Bottleneck(1024, 2048, 4, 4)
        self.bottleneck2 = Bottleneck(2048, 2048, 4, 4, False)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck2(x)
        return x


model = DeepLabV2(2)
image = torch.randn(1, 3, 224, 224)
# labels = torch.randint(0, 2, (1, 224, 224))
out = model(image)
print(out.shape)
# crf = CRF(2)
# out = out.permute(0, 2, 3, 1)
# out = out.reshape(1, -1, 2)
# labels = labels.reshape(1, -1)
# loss = -1 * crf(out, labels)
# print(loss)
