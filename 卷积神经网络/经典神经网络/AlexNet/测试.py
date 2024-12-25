import os

import torch
from torchvision.transforms import Compose, ToTensor, Resize

from utils.train_utils import load
from AlexNet import AlexNet
from PIL import Image

# 加载模型
result = load('./weights', AlexNet, {'num_classes': 3, 'dropout': 0.5})
model = result[0]
model.eval()

img_inputs = []

transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True)
])

# 加载测试数据
entries = os.scandir('./test_data')
for entry in entries:
    img_entries = os.scandir(entry.path)
    for img_entry in img_entries:
        print(img_entry.path)
        image = Image.open(img_entry.path)
        image = transform(image)
        img_inputs.append(image)

# 将列表转换成张量
img_inputs = torch.stack(img_inputs)

# 模型预测结果
with torch.inference_mode():
    y = model(img_inputs)
print(y)
values, indices = torch.topk(y, 1, dim=1)
indices = indices.squeeze(1)
print(indices)

fruit_dict = {
    0: 'Banana',
    1: 'Corn',
    2: 'Watermelon'
}

# indices.numpy() 获取张量对应的 numpy 数据
indices = indices.numpy()
result = [fruit_dict[indices[i]] for i in range(indices.shape[0])]
print(result)
