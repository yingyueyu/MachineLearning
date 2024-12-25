import torch
from torch import nn

x = torch.randn(5, 64, 100, 100)

conv3x3 = nn.Conv2d(64, 64, 3, bias=False)

conv5x5 = nn.Conv2d(64, 64, 5, bias=False)

# 使用3x3卷积
y = conv3x3(conv3x3(x))
print(y.shape)

# 使用5x5卷积
y = conv5x5(x)
print(y.shape)

# nn.Conv2d, nn.Linear 这些 nn 包中的模块，他们都继承了 nn.Module
# nn.Module 中有以下方法
# 1. parameters(): 获取模块内的所有参数
# 2. named_parameters(): 获取模块内所有参数的名称和参数值
# 3. named_modules(): 获取模块内的所有子模块

# 1. parameters()
print(conv3x3.parameters())
for param in conv3x3.parameters():
    print(param)
    print(type(param))  # => torch.nn.parameter.Parameter
    # torch.nn.parameter.Parameter 类型通常出现在 nn.Module 内部，作为注册到 nn.Module 中的参数使用，此处注册的目的是为了追踪梯度
    print(param.data)  # => tensor: 参数数据本身，不追踪梯度

# 2. named_parameters()
print(conv3x3.named_parameters())
for name, param in conv3x3.named_parameters():
    print(name)
    print(param)

# 3. named_modules()
print(conv3x3.named_modules())
for name, module in conv3x3.named_modules():
    print(name)
    print(module)

# 统计3x3和5x5的权重数量
print(sum([p.numel() for p in conv3x3.parameters()]) * 2)
print(sum([p.numel() for p in conv5x5.parameters()]))
print(conv3x3.weight.numel() * 2)
print(conv5x5.weight.numel())
