import torch
from torch import nn


# nn.GRUCell
# nn.GRU


# 采用 GRUCell 的模型
class CardModel_GRUCell(nn.Module):
    def __init__(self, input_size=19, hidden_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    # x 形状 (L=2, N, input_size=19)
    # h 形状 (N, hidden_size)
    def forward(self, x, h=None):
        L, N, input_size = x.shape
        # 初始化隐藏状态
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        # 更新隐藏状态
        for i in range(L):
            h = self.cell(x[i], h)
        # 输出
        y = self.fc(h)
        y = self.sigmoid(y)
        return y, h


# 采用 GRU 的模型
class CardModel_GRU(nn.Module):
    def __init__(self, input_size=19, hidden_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3, dropout=0.2)
        self.fc_out = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    # x 形状 (L=2, N, input_size=19)
    # h 形状 (D=1*num_layers=3, N, hidden_size)
    def forward(self, x, h=None):
        L, N, input_size = x.shape
        # 初始化 h
        if h is None:
            h = torch.zeros(3, N, self.hidden_size, device=x.device)
        # 创建 0 张量作为添加的最后一个输入，形状为 (1, N, input_size)
        zero = torch.zeros(1, N, input_size, device=x.device)
        # 连接张量
        x = torch.cat([x, zero], dim=0)
        # y 形状 (L=3, N, D=1*hidden_size=30)
        y, h = self.gru(x, h)
        # 输出
        y = self.fc_out(y[-1])
        y = self.sigmoid(y)
        return y, h


if __name__ == '__main__':
    model = CardModel_GRU()
    x = torch.randn(2, 5, 19)
    y, h = model(x)
    print(y)
    print(h.shape)
