import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc_wz = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_wr = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_w = nn.Linear(input_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        # 1. 计算更新门
        z_t = self.sigmoid(self.fc_wz(x) + self.fc_uz(h))
        # 2. 计算重置门
        r_t = self.sigmoid(self.fc_wr(x) + self.fc_ur(h))
        # 3. 计算候选隐藏状态
        h_t = self.tanh(self.fc_u(r_t * h) + self.fc_w(x))
        # 4. 更新隐藏状态
        h = z_t * h + (1 - z_t) * h_t
        return h


# 多对多GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)

    # x 形状 (L, N, input_size)
    # h 形状 (N, hidden_size)
    def forward(self, x, h=None):
        L, N, input_size = x.shape
        # 初始化 h
        if h is None:
            h = torch.zeros(N, self.hidden_size)

        outputs = []

        # 循环
        for i in range(L):
            h = self.cell(x[i], h)
            y = self.fc_out(h)
            outputs.append(y)

        outputs = torch.stack(outputs)

        return outputs, h


if __name__ == '__main__':
    # cell = GRUCell(2, 10)
    # x = torch.randn(5, 2)
    # h = torch.zeros(5, 10)
    # h = cell(x, h)
    # print(h.shape)

    model = GRU(2, 10)
    x = torch.randn(3, 5, 2)
    y, h = model(x)
    print(y.shape, h.shape)
    y, h = model(y, h)
    print(y.shape, h.shape)
