import json

import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from 数字识别模型 import NumberRecognition


def save(_model, _best_loss):
    state_dict = _model.state_dict()
    torch.save(state_dict, 'weights/model.pt')
    with open('weights/train_meta.json', 'w') as file:
        json.dump({'best_loss': _best_loss}, file)


def load():
    _model = NumberRecognition()
    _best_loss = float('inf')
    try:
        state_dict = torch.load('weights/model.pt')
        _model.load_state_dict(state_dict)
        with open('weights/train_meta.json', 'r') as file:
            _best_loss = json.load(file)['best_loss']
        print('加载模型成功')
    except:
        print('未找到预训练模型，初始化空模型')
    return _model, _best_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 10
lr = 1e-3
batch_size = 100

# 图片大小 28x28
ds = MNIST(root='./data', train=True, download=True, transform=ToTensor())
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

model, best_loss = load()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0


def train():
    global total_loss, count
    print('train start')
    model.train()
    for i, (inputs, labels) in enumerate(dl):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
        if (i + 1) * batch_size % 1000 == 0:
            print(f'batch: {i + 1}, data: {(i + 1) * batch_size}, avg loss: {total_loss / count}')
    print(f'train over; avg loss: {total_loss / count}')


for i in range(EPOCH):
    print(f'EPOCH: [{i + 1}/{EPOCH}]')
    train()

avg_loss = total_loss / count

if avg_loss < best_loss:
    save(model, avg_loss)
