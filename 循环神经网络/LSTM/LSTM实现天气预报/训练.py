import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from 循环神经网络.LSTM.LSTM实现天气预报.数据集 import WeatherDataset
from 循环神经网络.LSTM.LSTM实现天气预报.模型 import WeatherModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 1000
batch_size = 15
lr = 1e-2

# 数据集
ds = WeatherDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = WeatherModel()
model.to(device)

# 损失函数 优化器
optim = torch.optim.Adam(model.parameters(), lr=lr)

mse = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()


def loss_fn(y, label):
    l1 = mse(y[:, :2], label[:, :2])
    l2 = cross_entropy(y[:, 2:5], label[:, 2:5])
    l3 = cross_entropy(y[:, 5:], label[:, 5:])
    return l1 + l2 + l3


total_loss = 0.
count = 0

model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(dl):
        inputs, labels = inputs.transpose(0, 1).to(device), labels.to(device)
        optim.zero_grad()
        y, h, c = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optim.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

print('训练结束')

torch.save(model.state_dict(), 'weights/model.pt')
