import torch
from torch import nn


class LotteryModel(nn.Module):
    def __init__(self, input_size=17, hidden_size=30):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 多对多LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=3)
        # 全连接，将隐藏层长度转换成输入长度
        self.fc_out = nn.Linear(hidden_size, input_size)

    # x 形状 (N, 期数=5, L=8, input_size=17)
    # h 形状 (N, hidden_size)
    # c 形状 (N, hidden_size)
    def forward(self, x, h=None, c=None):
        N, T, L, input_size = x.shape
        # 初始化
        if h is None:
            h = torch.zeros(3, N, self.hidden_size, device=x.device)
        if c is None:
            c = torch.zeros(3, N, self.hidden_size, device=x.device)

        outputs = []

        # 按照期数循环
        for i in range(T):
            # 取出一期的8个中奖号码
            _x = x[:, i]
            # 用上面的一期数据预测下一期的8个中奖号码
            y, (h, c) = self.lstm(_x, (h, c))
            # 转换隐藏层长度为输出长度
            # 得到下一期预测的结果
            y = self.fc_out(y)
            outputs.append(y)

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1), h, c


if __name__ == '__main__':
    model = LotteryModel()
    x = torch.randn(6, 5, 8, 17)
    y, h, c = model(x)
    print(y.shape, h.shape, c.shape)
