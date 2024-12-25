# 流程:
# 1. 创建超参数
# 2. 创建数据集
# 3. 创建模型
# 4. 创建损失函数和优化器
# 5. 训练循环
#       1. 训练
#           1. 清空梯度
#           2. 前向传播
#           3. 计算损失
#           4. 反向传播
#           5. 更新参数
#       2. 验证
#           1. 前向传播
#           2. 计算损失
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Resize
from utils.train_utils import DatasetSpliter
from LeNet5 import LeNet5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 创建超参数
EPOCH = 100
batch_size = 100
lr = 1e-3

# 2. 创建数据集
transform = Compose([
    ToTensor(),
    Resize((32, 32), antialias=True)
])
ds = MNIST(root='./data', train=True, transform=transform, download=True)
ds_spliter = DatasetSpliter(ds, batch_size)

# 3. 创建模型
model = LeNet5()
model.to(device)
# 4. 创建损失函数和优化器
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

t_total_loss = 0.
t_count = 0


# 1. 训练
def train():
    global t_total_loss, t_count
    print('train start')
    model.train()
    for i, (inputs, labels) in enumerate(train_dl):
        inputs, labels = inputs.to(device), labels.to(device)
        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向传播
        y = model(inputs)
        # 3. 计算损失
        loss = loss_fn(y, labels)
        t_total_loss += loss.item()
        t_count += 1
        # 4. 反向传播
        loss.backward()
        # 5. 更新参数
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'batch: {i + 1}; avg loss: {t_total_loss / t_count}')
    print(f'train over; avg loss: {t_total_loss / t_count}')


v_total_loss = 0.
v_count = 0


# 2. 验证
def val():
    global v_total_loss, v_count
    print('val start')
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            # 1. 前向传播
            y = model(inputs)
            # 2. 计算损失
            loss = loss_fn(y, labels)
            v_total_loss += loss.item()
            v_count += 1
    avg_loss = v_total_loss / v_count
    print(f'val over; {avg_loss}')


# 5. 训练循环
for epoch in range(EPOCH):
    print(f'EPOCH: [{epoch + 1}/{EPOCH}]')
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    train()
    val()
