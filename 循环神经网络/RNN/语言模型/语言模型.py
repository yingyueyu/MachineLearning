import torch
from torch import nn


class LangModel(nn.Module):
    def __init__(self, hidden_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(16, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn2 = nn.RNN(16, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        # 输出层线性变换，将 hidden_size * 2 映射到 vocab_size=16
        self.fc_out = nn.Linear(hidden_size * 2, 16)
        # 学习权重参数
        self.fc_ex = nn.Linear(2, 32)
        self.fc_ex_re = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    # x 形状 (N, L, input_size=16)
    # h 形状 (2, D=2*num_layers=2, N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            # 初始化h
            h = torch.zeros(2, 4, N, self.hidden_size, device=x.device)

        # 调用循环神经网络
        # y1 形状 (N, L, hidden_size * 2)
        y1, h1 = self.rnn1(x, h[0])
        y2, h2 = self.rnn2(x, h[1])

        y1 = self.fc_out(y1)
        y2 = self.fc_out(y2)

        # 合并 y1 y2 两个输出
        # y 形状 (2, N, L, 16)
        y = torch.stack([y1, y2])
        # y 形状 (N, L, 16, 2)
        y = y.permute(1, 2, 3, 0)

        # 学习权重
        y = self.fc_ex(y)
        y = self.relu(y)
        y = self.fc_ex_re(y)
        # 激活 获得概率分布
        weights = self.softmax(y)

        # 求加权和
        y = y1 * weights[..., 0] + y2 * weights[..., 1]

        return y, h


if __name__ == '__main__':
    model = LangModel()
    x = torch.randn(8, 5, 16)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
