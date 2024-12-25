import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 遗忘门全连接
        self.fc_f = nn.Linear(input_size + hidden_size, hidden_size)
        # 输入门全连接
        self.fc_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc_i = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出门全连接
        self.fc_o = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # x 形状 (N, input_size)
    # h 形状 (N, hidden_size)
    # x 和 h 连接后的形状 (N, input_size + hidden_size)
    def forward(self, x, h, c):
        # 连接 x 和 h
        _x = torch.cat((x, h), dim=1)
        # 1. 遗忘门
        f_t = self.sigmoid(self.fc_f(_x))
        # 2. 输入门
        C_t = self.tanh(self.fc_c(_x))
        i_t = self.sigmoid(self.fc_i(_x))
        # 3. 更新长期记忆
        c = f_t * c + i_t * C_t
        # 4. 输出门
        o_t = self.sigmoid(self.fc_o(_x))
        h = o_t * self.tanh(c)
        return h, c


# 多对多LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        # 输出的全连接
        self.fc_out = nn.Linear(hidden_size, input_size)

    # x 形状 (L, N, input_size)
    # h 形状 (N, hidden_size)
    # c 形状 (N, hidden_size)
    def forward(self, x, h=None, c=None):
        L, N, input_size = x.shape
        # 初始化 h c
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        if c is None:
            c = torch.zeros(N, self.hidden_size)

        outputs = []

        # 循环，按照序列长度进行循环
        for i in range(L):
            # 更新长短期记忆
            h, c = self.cell(x[i], h, c)
            # 输出
            o = self.fc_out(h)
            outputs.append(o)

        # 输出形状 (L, N, input_size)
        outputs = torch.stack(outputs)
        return outputs, h, c


if __name__ == '__main__':
    # model = LSTMCell(2, 10)
    # x = torch.randn(3, 2)
    # h = torch.randn(3, 10)
    # c = torch.randn(3, 10)
    # h, c = model(x, h, c)
    # print(h.shape)
    # print(c.shape)

    model = LSTM(2, 10)
    x = torch.randn(3, 5, 2)
    y, h, c = model(x)
    print(y.shape)
    print(h.shape)
    print(c.shape)
    y, h, c = model(y, h, c)
    print(y.shape)
    print(h.shape)
    print(c.shape)

    # 固定随机数种子
    torch.manual_seed(100)
    # 声明输入数据和标签
    x = torch.randn(3, 5, 2)
    label = torch.randn(3, 5, 2)

    # 损失函数 优化器
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    EPOCH = 5000

    # 训练
    model.train()
    for epoch in range(EPOCH):
        optim.zero_grad()
        y, h, c = model(x)
        loss = loss_fn(y, label)
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {loss.item()}')
        loss.backward()
        optim.step()

    # 评估
    model.eval()
    with torch.inference_mode():
        y, h, c = model(x)

    # 是否近似相等
    print(torch.allclose(y, label, atol=0.01))
