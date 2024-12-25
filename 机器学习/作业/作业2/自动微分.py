# 1. 创建1张量 w，形状为 (1,)，并追踪其梯度
# 2. 创建0张量 b，形状为 (1,)，并追踪其梯度
# 3. 随机10组，输入x(10,)和标签y(10,)
# 4. 尝试对 w 的 [0] 元素进行赋值
# 5. 删除 4. 中的赋值代码并执行以下运算
#    $$
#    \_y = 3w^22x + 3b \\
#    loss = \frac{(\_y - y.mean())^2}{y.var()}
#    $$
# 6. 求 $\frac{\delta x}{\delta loss}$ 、$\frac{\delta b}{\delta loss}$、$\frac{\delta x}{\delta \_y}$、$\frac{\delta b}{\delta \_y}$
import torch

w = torch.ones(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

x = torch.randn(10)
y = torch.randn(10)

# print(w)
# print(w[0])
# w[0] = 2

# 执行函数追踪梯度
_y = torch.sum(3 * w ** 2 * 2 * x + 3 * b)
loss = (_y - y.mean()) ** 2 / y.var()

# 求导
# retain_graph=True 含义是，先反向传播并允许重画计算图（也就是允许重新进行反向传播）
_y.backward(retain_graph=True)
print(w.grad)
print(b.grad)


# 清空梯度
w.grad.zero_()
b.grad.zero_()

# 再次执行公式，重新追踪梯度
_y = torch.sum(3 * w ** 2 * 2 * x + 3 * b)
loss = (_y - y.mean()) ** 2 / y.var()

# 再次反向传播
loss.backward()
print(w.grad)
print(b.grad)
