import torch
import matplotlib.pyplot as plt

"""
1. 使用已讲的梯度下降方式求解，梯度最小的位置。
2. 绘制每一次梯度下降的点的位置（等高线图、3D图上进行标注）
"""

fig = plt.figure()
w1 = torch.linspace(-5, 5, 20)
w2 = torch.linspace(-5, 5, 20)

ax = fig.add_subplot(111, projection="3d")
w1, w2 = torch.meshgrid((w1, w2), indexing='ij')
loss = torch.cos(w1) * w1 + torch.cos(w2) * w2

ax.plot_surface(w1, w2, loss, cmap=plt.cm.YlGnBu_r, alpha=0.8)

# 动量法 + 加权平均位移
epochs = 20
points = []
w1_pre = torch.tensor(-2.5, requires_grad=True)
w2_pre = torch.tensor(-2.5, requires_grad=True)
v_t_w1 = 0
v_t_w2 = 0
beta = 0.1
for epoch in range(epochs):
    e = torch.cos(w1_pre) * w1_pre + torch.cos(w2_pre) * w2_pre
    e.backward()
    with torch.no_grad():
        slope_w1 = w1_pre.grad
        slope_w2 = w2_pre.grad
        ax.plot(w1_pre.detach().numpy(), w2_pre.detach().numpy(), e.detach().numpy(), 'ro--')
        # 动量法
        v_t_w1 = beta * v_t_w1 + slope_w1 * (1 - beta)
        v_t_w2 = beta * v_t_w2 + slope_w2 * (1 - beta)
        # v_t_w1 = v_t_w1 + slope_w1
        # v_t_w2 = v_t_w2 + slope_w2
        w1_pre -= 0.05 * v_t_w1
        w2_pre -= 0.05 * v_t_w2
        print(w1_pre, w2_pre)
        w1_pre.grad.zero_()
        w2_pre.grad.zero_()

plt.show()
