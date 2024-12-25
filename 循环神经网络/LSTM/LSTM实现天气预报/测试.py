import random

import torch

from 模型 import WeatherModel
from 数据集 import WeatherDataset

weather_map = {
    0: '晴',
    1: '阴',
    2: '雨',
}

wind_map = {
    0: '强',
    1: '中',
    2: '弱',
}


model = WeatherModel()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True, map_location='cpu'))
model.eval()

ds = WeatherDataset()
# 随机数据索引
idx = random.randint(0, len(ds) - 1)
print(idx)

for idx in range(len(ds)):
    inputs, label = ds[idx]
    inputs = inputs.unsqueeze(1)

    # 预测
    with torch.inference_mode():
        y, h, c = model(inputs)
        # (1, 8)

    print(y)
    print(f'模型预测结果: 天气[{weather_map[y[0][2:5].argmax().item()]}] 风力[{wind_map[y[0][5:].argmax().item()]}] 温度[{y[0][0].item():.2f}] 湿度[{y[0][1].item():.2f}]')
    print(f'真实数据结果: 天气[{weather_map[label[2:5].argmax().item()]}] 风力[{wind_map[label[5:].argmax().item()]}] 温度[{label[0].item():.2f}] 湿度[{label[1].item():.2f}]')

