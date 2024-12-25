import torch
import matplotlib.pyplot as plt
from torch import nn
from DenseUtil import CustomNet, train, accuracy

plt.subplot(121)
# 簇族A = 0
a = torch.normal(0.5, 0.4, (40, 2))
plt.plot(a[:, 0], a[:, 1], 'ro')
a_labels = torch.zeros((40, 1))
a = torch.concatenate((a, a_labels), dim=1)

# 簇族B = 1
b = torch.normal(1.5, 0.4, (40, 2))
plt.plot(b[:, 0], b[:, 1], 'bo')
b_labels = torch.ones((40, 1))
b = torch.concatenate((b, b_labels), dim=1)

# 簇族C = 2
c_x = torch.normal(0.5, 0.4, (40, 1))
c_y = torch.normal(1.5, 0.4, (40, 1))
c = torch.concatenate((c_x, c_y), dim=1)
plt.plot(c[:, 0], c[:, 1], 'go')
c_labels = torch.ones((40, 1)) * 2
c = torch.concatenate((c, c_labels), dim=1)

data = torch.concatenate((a, b, c), dim=0)
# torch中打乱所有点的顺序
indices = torch.randperm(data.shape[0])
data = data[indices]
x = data[:, :2]
y = data[:, 2]
y_copy = y.detach()

model = CustomNet(3)
loss = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(model.parameters(), lr=5e-2)
train(model, x, y, loss, sgd, 10000)
