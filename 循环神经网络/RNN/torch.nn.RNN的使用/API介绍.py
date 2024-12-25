import torch
import torch.nn as nn

# pytorch 官方 RNN 是一个多对多结构的 RNN
# 官方 API: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#rnn
# num_layers: 不同权重的RNN堆叠几层
# nonlinearity: 非线性激活函数
# bias: 是否学习偏置
# batch_first: 是否批次数，放到维度的首位
# dropout: dropout 操作，防止过拟合
# bidirectional: 是否是双向RNN
# 前两个参数为 input_size, hidden_size
model = nn.RNN(2, 10, num_layers=2, nonlinearity='tanh', bias=False, batch_first=True, dropout=0.4, bidirectional=True)

# 输入输出形状解释:
# 输入: input, h_0
# input: 输入参数，当只有一个批次数据时 (L, H_in)，否则为 (L, N, H_in)，当 batch_first=True 时，N 在最前面 (N, L, H_in)
# h_0: 输入隐藏状态，当只有一个批次数据时 (D * num_layers, H_out)，否则为 (D * num_layers, N, H_out)，当 batch_first=True 时，N 在最前面 (N, D * num_layers, H_out)
# 输出: output, h_n
# output: 输出参数，当只有一个批次数据时 (L, D * H_out)，否则为 (L, N, D * H_out)
# h_n: 输出隐藏状态，当只有一个批次数据时 (D * num_layers, H_out)，否则为 (D * num_layers, N, H_out)

# 对上述符号的解释如下:
# N: 批次数
# L: 序列长度
# D: 双向RNN时为2，否则为1
# H_in: 序列中，每个输入的长度
# H_out: 隐藏状态的长度

# 创建假数据
x = torch.arange(5 * 3 * 2, dtype=torch.float32).view(5, 3, 2)
labels = torch.tensor([
    [[1., 1.], [1., 1.], [1., 1.]],
    [[2., 2.], [2., 2.], [2., 2.]],
    [[3., 3.], [3., 3.], [3., 3.]],
    [[4., 4.], [4., 4.], [4., 4.]],
    [[5., 5.], [5., 5.], [5., 5.]]
])


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 官方API和nn.RNN相同
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, bias=True, batch_first=True, dropout=0.4,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        y = self.fc(y)
        y = self.relu(y)
        return y, h


model = RNN(2, 10)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for e in range(1000):
    optim.zero_grad()
    y, h = model(x)
    loss = loss_fn(y, labels)
    loss.backward()
    optim.step()
    if (e + 1) % 100 == 0:
        print(f'e: {e + 1}; loss: {loss.item()}')
        for weights in model.rnn.all_weights:
            for weight in weights:
                print(weight.grad.norm())

model.eval()
with torch.no_grad():
    y, h = model(x)
    print(y)
