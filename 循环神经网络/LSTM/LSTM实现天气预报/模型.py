import torch
from torch import nn


# 多对一的天气预测模型
class WeatherModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, h=None, c=None):
        L, N, input_size = x.shape
        # 初始化长短期记忆
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=x.device)
        if c is None:
            c = torch.zeros(N, self.hidden_size, device=x.device)

        # 循环输入序列
        for i in range(L):
            h, c = self.cell(x[i], (h, c))

        # 输出
        y = self.fc_out(h)

        return y, h, c


if __name__ == '__main__':
    model = WeatherModel()
    x = torch.randn(5, 16, 8)
    y, h, c = model(x)
    print(y.shape, h.shape, c.shape)
    print(y)
