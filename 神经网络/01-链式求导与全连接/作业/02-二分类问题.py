import torch
import matplotlib.pyplot as plt
from torch import nn

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

data = torch.concatenate((a, b), dim=0)
# torch中打乱所有点的顺序
indices = torch.randperm(data.shape[0])
data = data[indices]
x = data[:, :2]  # (40,2)
y = data[:, 2].reshape(-1, 1)  # (40,1)

# 构建模型
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.Tanh(),
    nn.Linear(20, 1),
    nn.Sigmoid()
)
# --- 损失函数 ---
criterion = nn.MSELoss()
# --- 优化器 ---
sgd = torch.optim.SGD(model.parameters(), lr=5e-3)
# --- 训练 ----
epochs = 1000
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
    if epoch % 100 == 0:
        print(f"epoch {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")

y_predict = model(x)
y_predict[y_predict >= 0.5] = 1
y_predict[y_predict < 0.5] = 0

# 准确率： 正确的样本在整体中的出现的概率
acc = sum(y_predict == y) / y.shape[0]
print(acc)

# 测试样本
x1_test = torch.linspace(0, 2, 20)
x2_test = torch.linspace(0, 2, 20)
x1_test, x2_test = torch.meshgrid((x1_test, x2_test), indexing='ij')
x1_test = x1_test.reshape(-1, 1)
x2_test = x2_test.reshape(-1, 1)
test = torch.concatenate((x1_test, x2_test), dim=1)
y_test = model(test)
y_test[y_test >= 0.5] = 1
y_test[y_test < 0.5] = 0
y_test = y_test.reshape(-1)
plt.subplot(122)
for i, (x, y) in enumerate(test):
    color = 'ro'
    if y_test[i] == 1:
        color = 'bo'
    plt.plot([x], [y], color)

plt.show()