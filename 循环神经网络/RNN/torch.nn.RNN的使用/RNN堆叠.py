# nn.RNN 参数中 num_layers 代表不同权重的 RNN 进行堆叠，堆叠时，将上一层 RNN 输出作为下一次的输入
# 此处重点演示不同的 RNN 堆叠方案
#       多层输出的合并方式:
#       1. 相加 相乘
#       2. 连接
#       3. 加权求和
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, bias=False)
        self.rnn2 = nn.RNN(input_size, hidden_size, bias=False)
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, h=None):
        L, N, input_size = x.shape

        if h is None:
            h = torch.zeros(2, N, self.hidden_size)

        # 分别执行两组 RNN
        y1, h1 = self.rnn1(x, h[0].unsqueeze(0))
        y2, h2 = self.rnn2(x, h[1].unsqueeze(0))

        # 堆叠隐藏状态
        h = torch.cat([h1, h2], dim=0)

        # 1. 相加
        # y = y1 + y2
        # 2. 连接
        # y = torch.cat([y1, y2], dim=-1)
        # 3. 加权求和
        # 求均值
        m1 = y1.view(y1.shape[0] * y1.shape[1] * y1.shape[2]).mean()
        m2 = y2.view(y2.shape[0] * y2.shape[1] * y2.shape[2]).mean()
        # 求权重
        w = torch.tensor([m1, m2])
        # 通过全连接学习权重
        w = self.fc1(w)
        w = self.fc2(w)
        # softmax 激活获得概率分布
        w = self.softmax(w)
        # 加权求和
        y = y1 * w[0] + y2 * w[1]
        return y, h


if __name__ == '__main__':
    model = RNN(2, 10)
    x = torch.randn(3, 5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
