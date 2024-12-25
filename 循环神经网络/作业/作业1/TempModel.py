from torch import nn


# 气温模型，是一个RNN模型，解决时序任务
class TempModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True, bidirectional=True, bias=False)
        # 声明一个全连接将输出的隐藏状态长度转换成气温特征长度，也就是1
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        y = self.fc(y)
        y = self.relu(y)
        return y, h
