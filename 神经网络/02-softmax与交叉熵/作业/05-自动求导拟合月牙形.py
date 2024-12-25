import torch
from torch import nn
import matplotlib.pyplot as plt


def one_hot(x):
    a = torch.zeros((2,))
    a[x] = 1
    return a


x = torch.arange(0, 10, 1)
y = torch.arange(0, 10, 1)
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

labels = [one_hot(z[i, j]) for i in range(10) for j in range(10)]
labels = torch.stack(labels, dim=0)

x1, x2 = torch.meshgrid([x, y], indexing="ij")
x1 = x1.unsqueeze(dim=-1)
x2 = x2.unsqueeze(dim=-1)
features = torch.concatenate([x1, x2], dim=-1)

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
    nn.Softmax(dim=-1)
)
ce_loss = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.05)

epochs = 5000
for epoch in range(epochs):
    sgd.zero_grad()
    features = features.reshape(-1, 2)
    predict = model(features.float())
    result = torch.mean(-labels * torch.log(predict))
    print(result.item())
    result.backward()
    sgd.step()

predict = model(features.float())
out = torch.argmax(predict, dim=1)
out = out.reshape(10, 10)

plt.subplot(121)
plt.contourf(x, y, z, levels=[0.5, 1], alpha=0.8, cmap='viridis')
plt.subplot(122)
plt.contourf(x, y, out, levels=[0.5, 1], alpha=0.8, cmap='viridis')
plt.show()
