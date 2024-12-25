# 官方的 torch.nn.RNN 就是多对多结构
# 每个输入对应一个输出，如图:
# y1    y2    y3
#  |     |     |
# s1 -> s2 -> s3
#  |     |     |
# x1    x2    x3
# 此处我们的重点是制作另一种多对多结构，如图:
#             y1    y2    y3
#              |     |     |
# s1 -> s2 -> s3 -> s4 -> s5
#  |     |     |
# x1    x2    x3
# 输出的条件，一般有两种
# 1. 输出到规定值，跳出循环
# 2. 输出长度达到上限，跳出循环

import torch
import torch.nn as nn


class ManyToMany_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_iter=4, break_value=None):
        super().__init__()
        self.max_iter = max_iter
        self.break_value = break_value
        self.cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    # x 形状 (L, N, input_size)
    def forward(self, x, h=None):
        L, N, input_size = x.shape

        outputs = []

        for i in range(L):
            h = self.cell(x[i], h)

        # 设置零张量作为输入
        x = torch.zeros_like(x[0])

        # 第一个输出
        y = self.fc(h)
        y = self.relu(y)
        outputs.append(y)

        if self.break_value is None or y != self.break_value:
            # 循环输出后续的的结果
            for i in range(self.max_iter - 1):
                h = self.cell(x, h)
                y = self.fc(h)
                y = self.relu(y)
                outputs.append(y)
                if self.break_value is not None and y == self.break_value:
                    break

        return torch.stack(outputs), h


if __name__ == '__main__':
    model = ManyToMany_RNN(2, 10)
    x = torch.randn(3, 5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
