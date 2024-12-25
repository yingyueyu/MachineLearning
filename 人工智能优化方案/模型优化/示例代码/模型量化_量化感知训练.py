import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 模型包含两个模块
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


# 初始化模型
model = SimpleModel()

# 选择量化配置
qat_config = torch.quantization.get_default_qat_qconfig('fbgemm')

# 融合线性层和ReLU层
# 融合结果成为一个新的层
model = torch.quantization.fuse_modules(model, ['linear', 'relu'])

# 设置量化配置
model.qconfig = qat_config

# 准备模型进行量化感知训练
torch.quantization.prepare_qat(model, inplace=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(2, 10)
y = torch.randn(2, 5)

# 训练模型
num_epochs = 10
# 训练循环内容已经是伪量化环境了
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 前向传播
    _y = model(x)
    loss = criterion(_y, y)
    # 反向传播和优化
    loss.backward()
    optimizer.step()

# 转换模型为量化模型，接下来模型就可以用于推理了
# 也不一定非要在这里转换，可以保存伪量化环境的模型model，等需要转换时再转换
torch.quantization.convert(model, inplace=True)

# 检查量化模型
print(model)
