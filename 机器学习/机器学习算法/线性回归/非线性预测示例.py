import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

# 超参数
EPOCH = 30000
lr = 1e-3

x = np.arange(10)


def expect(x):
    return (1.2 * x) ** 2 + 5


labels = [i + (-3 + np.random.rand() * 6) for i in expect(x)]

fig, ax = plt.subplots()

ax.plot(x, expect(x), 'y--')
ax.scatter(x, labels, c='r')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 10)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.fc(x)
        return x


model = Model()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
labels = torch.tensor(labels, dtype=torch.float).reshape(-1, 1)

with torch.no_grad():
    # 画初始线
    line, = ax.plot(x, model(x), 'b')

total_loss = 0.
count = 0

model.train()
# 训练
for epoch in range(EPOCH):
    # 清空梯度
    grad = 0
    # 预测
    y = model(x)

    line.set_ydata(y.detach().numpy())
    fig.canvas.draw()
    plt.pause(0.00167)

    # 计算损失
    loss = loss_fn(y, labels)
    total_loss += loss.item()
    count += 1

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}; loss: {total_loss / count}')

print('训练结束')

model.eval()
y = model(x)

line.set_ydata(y.detach().numpy())

plt.show()  # 显示图像
