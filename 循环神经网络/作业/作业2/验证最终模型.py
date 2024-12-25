import os

import torch

from 模型 import FinalModel
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from 数据集 import Tokenizer

# 加载模型
model = FinalModel('weights/R.pt', 'weights/L.pt')
model.eval()

# 转换器
transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True)
])

imgs = []

# 准备数据
for entry in os.scandir('data'):
    for img_entry in os.scandir(entry.path):
        print(entry.name)
        img = Image.open(img_entry.path)
        img = transform(img)
        imgs.append(img)

imgs = torch.stack(imgs)

# 预测
with torch.inference_mode():
    y = model(imgs)

print(y)

# 分词器
t = Tokenizer()
result = [''.join(t.decode(y[i].numpy())) for i in range(y.shape[0])]

print(result)
