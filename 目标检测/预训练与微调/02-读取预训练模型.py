import torch
import torchvision
from torch import nn
import numpy as np

# 设置设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义预训练模型
model = torchvision.models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)

# # 第一步：读取当前模型参数
# model_dict = model.state_dict()
# # 第二步：读取预训练模型
# pretrained_dict = torch.load("best_acc_80%.pth", weights_only=False, map_location=device)
# # for k, v in pretrained_dict.items():
# #     print(k, " : ", v.shape)
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
# # 第三步：使用预训练的模型更新当前模型参数
# model_dict.update(pretrained_dict)
# # 第四步：加载模型参数
# model.load_state_dict(model_dict)
#
# print(model.eval())

model_dict = model.state_dict()
pretrained_dict = torch.load("best_acc_80%.pth", weights_only=True, map_location=device)
temp = {}
for k, v in pretrained_dict.items():
    try:
        if np.shape(model_dict[k]) == np.shape(v):
            temp[k] = v
    except:
        pass
model_dict.update(temp)
print(model.eval())
