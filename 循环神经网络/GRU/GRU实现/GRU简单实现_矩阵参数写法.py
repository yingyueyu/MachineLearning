import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 直接将 nn.Parameter 对象赋值给 self 的属性，那么 pytorch 会自动注册这个参数
        self.w_z = nn.Parameter(torch.randn(input_size, hidden_size))
        self.u_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.w_r = nn.Parameter(torch.randn(input_size, hidden_size))
        self.u_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.u = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.w = nn.Parameter(torch.randn(input_size, hidden_size))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # self.register_parameter('Wz', w_z)
        # self.register_parameter('Uz', u_z)
        # self.register_parameter('Wr', w_r)
        # self.register_parameter('Ur', u_r)
        # self.register_parameter('U', u)
        # self.register_parameter('W', w)

    def forward(self, x, h):
        # 1. 计算更新门
        z_t = self.sigmoid(x @ self.w_z + h @ self.u_z)
        # 2. 计算重置门
        r_t = self.sigmoid(x @ self.w_r + h @ self.u_r)
        # 3. 计算候选隐藏状态
        h_t = self.tanh((r_t * h) @ self.u + x @ self.w)
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
    cell = GRUCell(3, 10)
    x = torch.randn(5, 2)
    h = torch.zeros(2, 10)
    # h = cell(x, h)
    # print(h.shape)
    # for n, p in cell.named_parameters():
    #     print(n)
    #     print(p)

    x = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])

    label = torch.tensor([
        [float(i) for i in range(10)],
        [float(i) for i in range(10)]
    ])

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(cell.parameters(), lr=0.01)

    for epoch in range(100):
        optim.zero_grad()
        # 将一次循环的输出作为下一次循环的输入时，若执行反向传播，则会产生重复反向传播的 bug
        # 所以此处新增 h = torch.zeros(2, 10) 这句话，使每次调用 cell 是的 h 都是一个零张量，而不是上一轮的输出
        h = torch.zeros(2, 10)
        h = cell(x, h)
        loss = loss_fn(h, label)
        print(loss.item())
        loss.backward()
        optim.step()

# model = GRU(2, 10)
# x = torch.randn(3, 5, 2)
# y, h = model(x)
# print(y.shape, h.shape)
# y, h = model(y, h)
# print(y.shape, h.shape)
