import torch
import matplotlib.pyplot as plt

x1 = torch.linspace(0, 1, 5)
x2 = torch.linspace(0, 1, 5)
x1, x2 = torch.meshgrid((x1, x2), indexing='ij')

z = torch.tensor([
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1]
])

plt.subplot(121)
plt.contour(x1, x2, z, levels=[0.5, 1], alpha=0.3, cmap='viridis')
plt.subplot(122)
plt.contourf(x1, x2, z, levels=[0.5, 1], alpha=0.3, cmap='viridis')
plt.show()
