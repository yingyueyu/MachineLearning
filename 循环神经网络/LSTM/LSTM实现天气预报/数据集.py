# 天气数据包含以下内容
# 1. 天气: ['晴', '阴', '雨']
# 2. 温度: 浮点数，10~30
# 3. 湿度: 40%~99%
# 4. 风力: ['强', '中', '弱']
import torch
from torch.utils.data import Dataset


# 创建 20 天的天气
# 0   晴  28.420107  89.009234  弱
# 1   晴  28.209043  88.269873  弱
# 2   阴  28.042489  84.559650  弱
# 3   雨  27.990343  77.531040  弱
# 4   晴  27.261524  76.583594  弱
# 5   阴  24.590137  72.673511  弱
# 6   雨  23.813508  72.011737  弱
# 7   阴  23.295155  71.711168  中
# 8   阴  23.090759  70.630823  中
# 9   雨  22.165793  61.657162  中
# 10  阴  20.736994  53.903101  中
# 11  晴  20.091849  52.600622  中
# 12  阴  19.111584  51.028392  中
# 13  晴  16.204785  48.898235  强
# 14  雨  16.022256  46.925580  强
# 15  阴  15.282910  45.722305  强
# 16  雨  15.276664  45.060256  强
# 17  雨  14.106988  44.293497  强
# 18  雨  13.117084  43.492269  强
# 19  晴  10.292210  41.945931  强

class WeatherDataset(Dataset):
    def __init__(self, C=5):
        super().__init__()
        self.data = [
            [0, 28.420107, 89.009234, 2],
            [0, 28.209043, 88.269873, 2],
            [1, 28.042489, 84.559650, 2],
            [2, 27.990343, 77.531040, 2],
            [0, 27.261524, 76.583594, 2],
            [1, 24.590137, 72.673511, 2],
            [2, 23.813508, 72.011737, 2],
            [1, 23.295155, 71.711168, 1],
            [1, 23.090759, 70.630823, 1],
            [2, 22.165793, 61.657162, 1],
            [1, 20.736994, 53.903101, 1],
            [0, 20.091849, 52.600622, 1],
            [1, 19.111584, 51.028392, 1],
            [0, 16.204785, 48.898235, 0],
            [2, 16.022256, 46.925580, 0],
            [1, 15.282910, 45.722305, 0],
            [2, 15.276664, 45.060256, 0],
            [2, 14.106988, 44.293497, 0],
            [2, 13.117084, 43.492269, 0],
            [0, 10.292210, 41.945931, 0],
        ]

        self.inputs = []
        self.labels = []

        iter_count = len(self.data) - C
        for i in range(iter_count):
            inp = self.data[i:i + C]
            label = self.data[i + C]
            self.inputs.append(inp)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        label = self.labels[idx]
        inp = torch.tensor([self._transform(d) for d in inp], dtype=torch.float)
        label = torch.tensor(self._transform(label), dtype=torch.float)
        return inp, label

    def _transform(self, weather_data):
        b1 = weather_data[1: 3]
        b2 = [1 if weather_data[0] == i else 0 for i in range(3)]
        b3 = [1 if weather_data[3] == i else 0 for i in range(3)]
        return b1 + b2 + b3


if __name__ == '__main__':
    ds = WeatherDataset()
    print(len(ds))
    print(ds[0])
