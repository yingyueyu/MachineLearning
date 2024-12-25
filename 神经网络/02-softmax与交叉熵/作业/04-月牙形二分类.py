import torch
from torch import nn
import matplotlib.pyplot as plt
from DenseUtil import CustomNet, train


# 独热编码
def one_hot(x):
    a = torch.zeros((2,))
    a[x] = 1
    return a


# 特征输入
x = torch.arange(0, 10, 1)
y = torch.arange(0, 10, 1)
# 真实类别
z = torch.tensor([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

x, y = torch.meshgrid((x, y), indexing='ij')
# x (10,10) => x (10,10,1)
x = torch.unsqueeze(x, -1)
y = torch.unsqueeze(y, -1)
# 输入特征
features = torch.concatenate((x, y), dim=-1)

# ---- 训练模型 ----
model = CustomNet(2)
criterion = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.05)

# 测试 100,2 作为输入特征的情况
features = features.reshape(-1, 2)
train(model, features.float(), z.reshape(-1), criterion, sgd, 5000)

features = features.reshape(10, 10, 2)
plt.subplot(121)
plt.contourf(features[:,:, 0], features[:,:, 1], z, levels=[0.5, 1], alpha=0.3, cmap='viridis')
plt.subplot(122)
predict = model(features.float())
predict = torch.argmax(predict, dim=-1).reshape(10, 10)
plt.contourf(features[:,:, 0], features[:,:, 1], predict, levels=[0.5, 1], alpha=0.3, cmap='viridis')
plt.show()
