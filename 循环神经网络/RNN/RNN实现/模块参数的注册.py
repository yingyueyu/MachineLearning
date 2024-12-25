# 模块参数的注册
# 可以在一个模块内将一个变量，注册称为这个模块的参数(Parameters)
# 这些参数是一些可以被学习的参数，模型可以通过 model.parameters() 获取参数
import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc = nn.Linear(10, 10)
        # self.w = torch.randn(10, 10)
        # 使用 nn.Parameter 创建参数
        # self.w = nn.Parameter(torch.randn(10, 10))

        _w = torch.randn(10, 10)
        # 注册参数
        # 第一个参数: 参数名
        # 第二个参数: 参数值
        self.register_parameter('weight', nn.Parameter(_w))

    def forward(self, x):
        x @ self.w


if __name__ == '__main__':
    model = MyModule()
    # for n, p in model.named_parameters():
    #     print(n)
    #     print(p)

    print(model.weight)
