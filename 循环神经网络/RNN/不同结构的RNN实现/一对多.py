import torch
from torch import nn


# 在一对多的前向传播中会输出多个值
# 但是应该输出多少个值呢？
# 所以当输出数量不固定时，我们需要一个跳出输出的条件，一般有两种
# 1. 输出到规定值，跳出循环
# 2. 输出长度达到上限，跳出循环
# 所以我们可以在创建模型时设置跳出循环的条件

class OneToMany_RNN(nn.Module):
    # max_iter: 最大循环次数
    # break_value: 跳出循环的值
    def __init__(self, input_size, hidden_size, max_iter=3, break_value=None):
        super().__init__()
        self.max_iter = max_iter
        self.break_value = break_value
        self.cell = nn.RNNCell(input_size, hidden_size)
        # 输出层全连接
        self.fc = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    # x 形状: (N, input_size)
    def forward(self, x, h=None):
        outputs = []
        for i in range(self.max_iter):
            # h (N, hidden_size)
            h = self.cell(x, h)
            # 输出一个循环的结果
            y = self.fc(h)
            y = self.relu(y)
            outputs.append(y)
            if i == 0:
                # 第一次输入完成后，将输入x转换成零张量
                x = torch.zeros_like(x)

            # 判断是否满足跳出循环的条件
            if self.break_value is not None and y == self.break_value:
                break
        return torch.stack(outputs), h


if __name__ == '__main__':
    model = OneToMany_RNN(2, 10)
    x = torch.randn(5, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
