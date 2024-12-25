import torch
from torch import nn


class WeatherModel(nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(8, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.RNN(8, hidden_size, bidirectional=True, batch_first=True)
        # 将 rnn 的输出转换成我们希望的 8 个值
        self.fc = nn.Linear(hidden_size * 2, 8)
        self.fc_exp = nn.Linear(2, 10)
        self.fc_exp_re = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=-1)

    # x 形状 (N, L, input_size=8)
    # h 形状 (2, D=2 * num_layers=1, N, hidden_size=10)，因为我们有两个两个rnn，所以 h 也有两组
    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(2, 2, x.shape[0], self.hidden_size)

        # 将输入序列 x 分别输入到 rnn1 和 rnn2 中进行预测
        # y1 形状 (N, L=5, hidden_size * 2)
        y1, h1 = self.rnn1(x, h[0])
        y2, h2 = self.rnn2(x, h[1])
        # 线性变换转换形状
        # y1[:, -1] 形状 (N, hidden_size * 2)
        y1 = self.fc(y1[:, -1])
        y2 = self.fc(y2[:, -1])
        # 转换形状 (N, input_size, 2)
        y = torch.stack([y1, y2]).permute(1, 2, 0)

        # 借鉴序列激发的思想，执行以下内容
        # 使用线性变换学习 y1 和 y2 的权重
        y = self.fc_exp(y)
        y = self.fc_exp_re(y)
        # 使用 softmax 激活，得到权重值
        weight = self.softmax(y)

        # 加权求和
        y = y1 * weight[:, :, 0] + y2 * weight[:, :, 1]

        return y, h


if __name__ == '__main__':
    model = WeatherModel()
    x = torch.randn(6, 5, 8)
    y, h = model(x)
    print(y.shape)
