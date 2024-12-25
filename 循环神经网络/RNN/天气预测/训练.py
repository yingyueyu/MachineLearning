import torch
from torch import nn
from torch.utils.data import DataLoader

from 数据集 import WeatherDataset
from 天气预报模型 import WeatherModel

# 定义设备
# 两种情况需要设置设备
# 1. 模型需要迁移到设备上 model.to(device)
# 2. 参与运算的需要优化训练的数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 10000
batch_size = 15
lr = 0.001


# 数据集

# 数据编码函数
def data_encoder(d):
    s1 = d[1:3]
    # one-hot 编码
    s2 = [1 if i == d[0] else 0 for i in range(3)]
    s3 = [1 if i == d[3] else 0 for i in range(3)]
    return s1 + s2 + s3


def transform(inp):
    r = []
    for i in range(len(inp)):
        d = inp[i]
        r.append(data_encoder(d))
    return torch.tensor(r)


def target_transform(label):
    return torch.tensor(data_encoder(label))


ds = WeatherDataset(transform=transform, target_transform=target_transform)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = WeatherModel()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
model.to(device)

mse = nn.MSELoss()
bce = nn.BCELoss()
sigmoid = nn.Sigmoid()


# 损失函数 优化器
# _y: 模型预测的结果
def loss_fn(_y, label):
    # _y 形状 (N, 8)
    # label 形状 (N, 8)
    # 气温和湿度产生的损失
    l1 = mse(_y[:, :2], label[:, :2])
    # 天气产生的损失
    l2 = bce(sigmoid(_y[:, 2:5]), label[:, 2:5])
    # 风力产生的损失
    l3 = bce(sigmoid(_y[:, 5:]), label[:, 5:])
    return l1 + l2 + l3


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

# 训练
model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(dl):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y, h = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
