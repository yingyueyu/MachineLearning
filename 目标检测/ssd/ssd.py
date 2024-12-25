import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT, num_classes=2)

# 测试模型
# model.eval()
# x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
# predictions = model(x)
# print(predictions) # 首先返回一个列表，一张图片对应一张列表，每个元素有boxes、scores，labels

# 模型的训练
model.train()
# SSD中关于tranning的部分 与 FasterRCNN 中的内容一模一样
# 检测汽车区域，要求针对VOC中的汽车的数据，进行预处理后，通过SSD进行训练，并且预测汽车的位置
