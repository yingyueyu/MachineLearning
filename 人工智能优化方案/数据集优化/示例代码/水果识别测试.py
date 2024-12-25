import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

from 水果分类器神经网络 import FruitClassifier

image_path = './data/Watermelon3.jpg'
state_dict_path = './weights/fruit_model.20.pt'
fruit_map = {
    0: '苹果',
    1: '香蕉',
    2: '西瓜'
}

image = Image.open(image_path)

transformer = Compose([
    ToTensor(),
    Resize((100, 100), antialias=True)
])

image = transformer(image)
image = torch.unsqueeze(image, 0)

model = FruitClassifier()
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    y = model(image)
    print(y)
    idx = torch.argmax(y)
    print(idx)
    print(fruit_map[idx.item()])
