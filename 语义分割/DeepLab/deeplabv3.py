import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50

# 可以通过torchvision.models.segmentation 直接使用deeplabv3 代码
# 预训练参数 DeepLabV3_ResNet50_Weights

# model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=False)
images = torch.randn(2, 3, 480, 480)
device = torch.device("cuda:0")
model = model.to(device)
result = model(images.to(device).float())
print(result['out'].shape)
