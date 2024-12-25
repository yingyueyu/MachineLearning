import torch
from torch import nn


class OneToOne_RNN(nn.Module):
    # output_size: 想要输出的特征长度
    # def __init__(self, input_size, hidden_size, output_size):
    # 通常 input_size 等于 output_size
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.RNNCell(input_size, hidden_size)
        # 输出层全连接
        self.fc = nn.Linear(hidden_size, input_size)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        h = self.cell(x, h)
        y = self.fc(h)
        y = self.relu(y)
        return y, h


if __name__ == '__main__':
    model = OneToOne_RNN(2, 10)
    x = torch.randn(5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
