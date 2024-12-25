import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


# 一阶段，图像识别模型
class ImageRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用官方的 mobilenetv3
        self.net = mobilenet_v3_small()
        # 将 1000 分类转换成 3 分类
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.net(x)
        y = self.fc(x)
        # 此处不激活，后续使用交叉熵损失函数
        return y


# 二阶段模型，语言模型
class LangModel(nn.Module):
    # max_iter: 最大循环次数
    # break_value_idx: 退出符号的索引
    def __init__(self, hidden_size=30, max_iter=5, break_value_idx=11):
        super().__init__()
        self.max_iter = max_iter
        self.break_value_idx = break_value_idx
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(1, hidden_size)
        # 全连接，输出 12 个字符的概率分布
        self.fc_out = nn.Linear(hidden_size, 12)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    # x 形状 (N, 1)
    # h 形状 (N, hidden_size)
    def forward(self, x, h=None):
        N, input_size = x.shape
        # 初始化 h
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=x.device)

        outputs = []

        for i in range(self.max_iter):
            # 更新隐藏状态
            h = self.cell(x, h)
            if i == 0:
                # 第一轮输出后就将x变成零
                x = torch.zeros_like(x)
            # 预测输出一个词
            out = self.fc_out(h)
            # 激活，得到概率分布
            out = self.log_softmax(out)
            outputs.append(out)

        y = torch.stack(outputs).transpose(0, 1)

        return y, h


# 合并两个阶段的最终模型
class FinalModel(nn.Module):
    # r_path: 一阶段模型的参数路径
    # l_path: 二阶段模型的参数路径
    def __init__(self, r_path, l_path, device='cpu'):
        super().__init__()
        self.R = ImageRecognition()
        state_dict = torch.load(r_path, weights_only=True, map_location=device)
        self.R.load_state_dict(state_dict)
        self.L = LangModel()
        state_dict = torch.load(l_path, weights_only=True, map_location=device)
        self.L.load_state_dict(state_dict)

    # x 形状 (N, 3, 224, 224)
    def forward(self, x):
        # 预测图片的分类
        # y 形状 (N, 3)
        y = self.R(x)
        # 求最大值索引
        idx = y.argmax(-1)
        idx = idx.reshape(idx.shape[0], 1).to(torch.float)
        # 预测语言
        y, h = self.L(idx)
        y = y.argmax(-1)
        return y


if __name__ == '__main__':
    # model = ImageRecognition()
    # x = torch.randn(5, 3, 224, 224)
    # y = model(x)
    # print(y)

    # model = LangModel()
    # x = torch.tensor([
    #     [0],
    #     [1]
    # ], dtype=torch.float)
    # print(x.shape)
    # y, h = model(x)
    # print(y.shape)

    model = FinalModel('./weights/R.pt', './weights/L.pt')
    x = torch.randn(5, 3, 224, 224)
    model(x)
