import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from bbox.selective_search import generate_bbox


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(512, 4096)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':
    # 此处选择性搜索算法的颜色标准是RGB
    image = cv2.imread("../img/catdog.jpg")
    h, w = image.shape[:2]
    anchors = generate_bbox(image)

    features = []
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        feature = image[y1:y2, x1:x2]
        feature = cv2.resize(feature, (224, 224))
        features.append(feature)

    features = np.stack(features, axis=0)
    features = torch.from_numpy(features).permute([0, 3, 1, 2])
    print(features.shape)

    resnet = ResNet()
    b = resnet(features.float())
    print(b.shape)
