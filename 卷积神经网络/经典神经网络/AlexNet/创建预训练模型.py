# pytorch 中有专门的 AlexNet 预训练模型，通过如下方式创建:
# torchvision.models.alexnet，该函数返回一个 AlexNet 类
import os

import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

# 直接创建 AlexNet 模型
# api: torchvision.models.alexnet(weights, progress) -> AlexNet
# weights: 可选参数，是否采用预定义权重，默认不采用。若想使用默认权重，可以看这个: https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.AlexNet_Weights
# progress: 默认为 True，是否显示下载参数的进度
model = torchvision.models.alexnet(weights='DEFAULT', progress=True)
print(model)
model.eval()

# 创建未训练的空模型
# model = torchvision.models.AlexNet()
# print(model)

transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True),
])

imgs = []

entries = os.scandir('../ImageNet_validation_images')
for entry in entries:
    image = Image.open(entry.path)
    imgs.append(transform(image))

imgs = torch.stack(imgs)

# 这里用官方预训练模型的话，输出分类数量高达 1000，因为该模型基于 ImageNet 数据集训练的
# label 对照表 https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
with torch.inference_mode():
    y = model(imgs)
values, indices = torch.topk(y, 5)
print(indices)
