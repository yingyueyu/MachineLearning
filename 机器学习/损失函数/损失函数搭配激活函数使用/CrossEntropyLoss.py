# CrossEntropyLoss 交叉熵损失函数
# 对应激活函数: softmax
# 应用场景: 多用于多分类问题求损失
# CrossEntropyLoss 其实就是 log_softmax 和 NLLLoss 的组合


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
# y = nn.Sigmoid()(y)
y = nn.functional.softmax(y, dim=-1)
# print(y)
# 计算损失
# 交叉熵损失函数中已经内置了 log_softmax 进行激活，所以此处不需要使用 softmax 进行激活
# loss_fn = nn.BCELoss()
loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(y, torch.tensor([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ]))
loss = loss_fn(y, torch.tensor([0, 1, 2]))
print(loss)
