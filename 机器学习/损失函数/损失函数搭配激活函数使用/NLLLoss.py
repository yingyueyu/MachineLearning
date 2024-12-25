# NLLLoss 负对数似然损失函数
# 对应激活函数: log_softmax
# 应用场景: 用于多分类问题


import torch
import torch.nn as nn

torch.manual_seed(100)

# 假设 x 是三种交通工具的图片，分别是 汽车0 飞机1 火车2
x = torch.rand(3, 100, 100)
# 展平操作
flatten = nn.Flatten(start_dim=1)
x = flatten(x)
print(x.shape)

# Linear 输出 3，此处第二个参数 3 取决于分类个个数，例如此处交通工具分了 3 类
fc = nn.Linear(100 * 100, 3)

y = fc(x)
# 激活
y = nn.functional.log_softmax(y, dim=-1)
# print(y)
# 计算损失
loss_fn = nn.NLLLoss()
loss = loss_fn(y, torch.tensor([0, 1, 2]))
print(loss)
