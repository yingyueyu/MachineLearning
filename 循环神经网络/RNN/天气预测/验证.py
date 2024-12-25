import random

import torch

from 天气预报模型 import WeatherModel
from 数据集 import WeatherDataset

# 加载模型
model = WeatherModel()
# map_location: 将要加载的模型参数迁移到哪个设备上
state_dict = torch.load('weights/model.pt', weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # 开启评估模式


# 加载数据
# 数据编码函数
def data_encoder(d):
    s1 = d[1:3]
    # one-hot 编码
    s2 = [1 if i == d[0] else 0 for i in range(3)]
    s3 = [1 if i == d[3] else 0 for i in range(3)]
    return s1 + s2 + s3


def transform(inp):
    r = []
    for i in range(len(inp)):
        d = inp[i]
        r.append(data_encoder(d))
    return torch.tensor(r)


def target_transform(label):
    return torch.tensor(data_encoder(label))


ds = WeatherDataset(transform=transform, target_transform=target_transform)
# # 随机索引
# idx = random.randint(0, len(ds) - 1)
# print(idx)
# # 取出数据
# inp, label = ds[idx]

for idx in range(len(ds)):
    inp, label = ds[idx]

    # 预测
    with torch.inference_mode():
        # inp 是二维数据，而训练模型用的是三维数据，所以此处使用 unsqueeze 增加一个批次维度
        y, h = model(inp.unsqueeze(0))

    y = y[0]

    print(y)
    print(label)

    weather_map = {
        0: '晴',
        1: '阴',
        2: '雨'
    }

    wind_map = {
        0: '强',
        1: '中',
        2: '弱'
    }

    print(
        f'模型预测结果: 温度[{y[0]:.2f}] 湿度[{y[1]:.2f}] 天气[{weather_map[y[2:5].argmax().item()]}] 风力[{wind_map[y[5:].argmax().item()]}]')
    print(
        f'真实数据结果: 温度[{label[0]:.2f}] 湿度[{label[1]:.2f}] 天气[{weather_map[label[2:5].argmax().item()]}] 风力[{wind_map[label[5:].argmax().item()]}]')
