import torch
from torch import nn
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 40)
y = 3 * x ** 3 - 2 * x ** 2 - x + 5

plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')

# --- 构建模型、损失函数、优化器（w，b更新）---
# --- 模型 ---
model = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)
# --- 损失函数 ---
criterion = nn.MSELoss()
# --- 优化器 ---
sgd = torch.optim.Adam(model.parameters(), lr=1e-3)
# 针对输入输出数据进行预处理
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
# --- 训练 ----
epochs = 10000
for epoch in range(epochs):
    # 清空w，b 的梯度
    sgd.zero_grad()
    # 预测
    y_predict = model(x)
    # 损失
    loss = criterion(y_predict, y)
    # 反向传播更新梯度
    loss.backward()
    sgd.step()
    if epoch % 1000 == 0:
        print(f"epoch {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")

y_predict = model(x)
plt.plot(x.detach().numpy(), y_predict.detach().numpy(), 'b--')
plt.show()
