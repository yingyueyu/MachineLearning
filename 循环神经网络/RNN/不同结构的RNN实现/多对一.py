import torch
from torch import nn


# 多对一中，RNNCell不直接输出结果，只有循环的最后次，需要输出结果

class ManyToOne_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    # x 形状 (L, N, input_size)
    def forward(self, x, h=None):
        L, N, input_size = x.shape
        # 循环更新隐藏状态，但不输出
        for i in range(L):
            # h 形状 (N, hidden_size)
            h = self.cell(x[i], h)
        # 最后次输出结果
        y = self.fc(h)
        y = self.relu(y)
        return y, h


if __name__ == '__main__':
    model = ManyToOne_RNN(2, 10)
    x = torch.randn(3, 5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
