# RNN 实现步骤
# 这里我们实现 pytorch 官方的 RNN 结构，多对多结构，让每个时间步都输出一个值
# 1. 实现 RNNCell
# 2. 实现 RNN
import torch
from torch import nn


# 循环神经网络单元
class RNNCell(nn.Module):
    # input_size: 输入长度
    # hidden_size: 隐藏层长度
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # hidden_state (          1, hidden_size)
        # w_hidden     (hidden_size, hidden_size)
        _w_hidden = torch.randn(hidden_size, hidden_size)
        # x            (         1, input_size)
        # w_input      (input_size, hidden_size)
        _w_input = torch.randn(input_size, hidden_size)

        self.register_parameter('w_hidden', nn.Parameter(_w_hidden))
        self.register_parameter('w_input', nn.Parameter(_w_input))

    # x 输入形状 (1, input_size)
    # N: 批次数
    # input_size: 输入特征数
    def forward(self, x, hidden_state=None):
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = torch.zeros(1, self.hidden_size)
        # 更新隐藏状态
        hidden_state = nn.functional.tanh(hidden_state @ self.w_hidden + x @ self.w_input)
        return hidden_state


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        # hidden_state (          1, hidden_size)
        # w_output     (hidden_size, hidden_size)
        _w_output = torch.randn(hidden_size, hidden_size)

        self.register_parameter('w_output', nn.Parameter(_w_output))

    # x 形状: (L, input_size)
    # L: 序列长度
    # input_size: 输入特征数
    def forward(self, x, hidden_state=None):
        L, input_size = x.shape
        # 输出列表
        outputs = []
        for i in range(L):
            # 调用循环单元
            hidden_state = self.cell(x[i], hidden_state)
            # 线性变换求输出
            output = hidden_state @ self.w_output
            # 加入输出列表
            outputs.append(output)

        # 输出2个元素
        # 1. 输出结果
        # 2. 隐藏状态
        return torch.stack(outputs).squeeze(1), hidden_state


if __name__ == '__main__':
    model = RNN(2, 10)
    # 第一个序列
    s1 = torch.randn(3, 2)
    y, hs = model(s1)
    print(y.shape)
    print(hs.shape)
    # 第二个序列
    s2 = torch.randn(3, 2)
    y, hs = model(s2, hs)
    print(y.shape)
    print(hs.shape)

