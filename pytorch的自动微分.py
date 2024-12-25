import torch
from matplotlib import pyplot as plt

# 总结
# 1. 给需要自动求梯度的张量添加 requires_grad=True
# 2. 反向传播（backward）后才会自动求得梯度
# 3. 若叶节点需要记录梯度时，叶节点不能执行就地操作
# 4. 可以使用 .detach() 将任意计算图中的张量克隆并脱离计算图
# 5. 反向传播得到的梯度会累加，大部分情况训练时需要清空梯度


# 超参数
EPOCH = 10000
lr = 1e-3

# 创建输入
x = torch.arange(10)


# 期望函数
def expect(x):
    return 1.5 * x - 7


# 真实标签
y = expect(x) + torch.normal(0, 2, x.shape)

# requires_grad = True 表示需要自动计算梯度
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 声明模型
def model(x):
    return w * x + b


# 损失函数
def MSELoss(_y, y):
    return torch.sum((_y - y) ** 2)


fig, ax = plt.subplots()
# 绘制样本点
ax.scatter(x.numpy(), y.numpy(), 64, c='r')
# 画期望直线
ax.plot(x, expect(x), 'y--')

# 此处不训练模型，所以不记录梯度
with torch.no_grad():
    # 绘制模型初始状态
    line, = ax.plot(x, model(x))

# 训练循环
for epoch in range(EPOCH):
    # 1. 清空梯度
    # 梯度是累加的，所以每次训练前需要清空梯度
    w.grad = None  # 张量中的 grad 属性保存的是他的导数
    b.grad = None
    # 2. 模型预测
    _y = model(x)

    # 更新直线图
    # _y.detach() 克隆张量并脱离计算图
    line.set_ydata(_y.detach())
    fig.canvas.draw()
    plt.pause(0.0167)

    # 3. 计算损失值
    loss = MSELoss(_y, y)
    # print(loss.item())
    # 4. 计算梯度 (反向传播)
    # 反向传播后就会计算出导数
    loss.backward()
    # 5. 更新参数
    # 添加上下文管理器，防止优化参数时去追踪参数的梯度
    with torch.no_grad():
        # 在此上下文范围中不会追踪梯度
        # 此处应当使用 -= 操作而不是 = 赋值操作
        # 张量在使用 -= 时是就地（inplace）赋值操作
        w -= lr * w.grad
        b -= lr * b.grad

plt.axis('equal')
plt.show()
