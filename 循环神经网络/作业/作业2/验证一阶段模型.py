import os

import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

from 模型 import ImageRecognition

# 加载模型
model = ImageRecognition()
state_dict = torch.load('weights/R.pt', weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True)
])

entries = os.scandir('data')

imgs = []

# 打开目录下的图片并存储到 imgs
for entry in entries:
    img_entries = os.scandir(entry.path)
    for img_entry in img_entries:
        image = Image.open(img_entry.path)
        image = transform(image)
        imgs.append(image)

# 堆叠
imgs = torch.stack(imgs)

with torch.inference_mode():
    y = model(imgs)

idx = y.argmax(dim=-1)

print(idx)
