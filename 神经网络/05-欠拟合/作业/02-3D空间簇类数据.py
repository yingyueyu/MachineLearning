import torch
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# 簇族A = 0
a = torch.normal(0.5, 0.3, (40, 3))
ax.plot(a[:, 0], a[:, 1], a[:, 2], 'ro')
a_labels = torch.zeros((40, 1))
a = torch.concatenate((a, a_labels), dim=1)

# 簇族B = 1
b = torch.normal(1.5, 0.3, (40, 3))
ax.plot(b[:, 0], b[:, 1], b[:, 2], 'bo')
b_labels = torch.ones((40, 1))
b = torch.concatenate((b, b_labels), dim=1)

# 簇族C = 2
c_x = torch.normal(0.5, 0.3, (40, 2))
c_y = torch.normal(1.5, 0.3, (40, 1))
c = torch.concatenate((c_x, c_y), dim=1)
ax.plot(c[:, 0], c[:, 1], c[:, 2], 'go')
c_labels = torch.ones((40, 1)) * 2
c = torch.concatenate((c, c_labels), dim=1)

data = torch.concatenate((a, b, c), dim=0)
# torch中打乱所有点的顺序
indices = torch.randperm(data.shape[0])
data = data[indices]

# torch.save 保存模型、保存参数（只要是torch.tensor都可以被保存）
torch.save(data, './data.pth')
