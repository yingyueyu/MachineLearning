import torch
from torch import nn, optim

# 梯度裁剪原理如下:

# 模拟梯度
grad = torch.tensor([1., 1.2, 5., 0.4, 0.8])
# 梯度范数
print(grad.norm())
# 定义裁剪阈值
threshold = 2
# 当大于阈值则裁剪
if grad.norm() > threshold:
    # 计算符合阈值的新的梯度
    grad = grad / grad.norm() * threshold
    print(grad)
    print(grad.norm())


# 使用官方 API 对 Module 实例进行裁剪

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 示例输入和目标
inputs = torch.randn(5, 10)
targets = torch.randn(5, 1)

# 定义损失函数
criterion = nn.MSELoss()

# 前向传播
outputs = model(inputs)
loss = criterion(outputs, targets)

# 反向传播
optimizer.zero_grad()
loss.backward()

# 梯度裁剪发生在反向传播后

# torch.nn.utils.clip_grad_norm_ 梯度裁剪
# 该梯度裁剪将以参数的梯度值作为参考，梯度值大于阈值将被裁剪

# max_norm = 2.0
# # 计算总梯度范数
# total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
# print(f'手动计算的总梯度范数: {total_norm}')
#
# # 使用torch.nn.utils.clip_grad_norm_计算并裁剪梯度
# clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
# print(f'clip_grad_norm_返回的总梯度范数: {clip_norm}')
#
# total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
# print(f'裁剪后，手动计算的总梯度范数: {total_norm}')


# torch.nn.utils.clip_grad_value_ 梯度裁剪
# 该梯度裁剪将以梯度张量内的值作为参考，张量中的值大于了阈值将被设置到阈值范围
# 如: 阈值为 0.5，则大于 0.5 的值将被设为 0.5，小于 -0.5 的值将被设为 -0.5

# 打印裁剪前的梯度
print("裁剪前的梯度:")
for p in model.parameters():
    print(p.grad)
    print(p.grad.norm())

# 梯度值裁剪
clip_value = 0.5
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

# 打印裁剪后的梯度
print("裁剪后的梯度:")
for p in model.parameters():
    print(p.grad)
    print(p.grad.norm())

# 参数更新
optimizer.step()

# 梯度剪枝
# 梯度剪枝很类似 torch.nn.utils.clip_grad_value_ 梯度裁剪
# 不过梯度剪枝在小于阈值时是直接设为 0
# 而 torch.nn.utils.clip_grad_value_ 是设为指定阈值

# 定义一个示例梯度张量
grad = torch.tensor([0.1, 0.3, 0.0, 0.4, 0.2])

# 剪枝阈值
threshold = 0.25

# 梯度剪枝：将小于阈值的梯度置为0
pruned_grad = grad.clone()
pruned_grad[pruned_grad.abs() < threshold] = 0

print("原始梯度:", grad)
print("剪枝后的梯度:", pruned_grad)
