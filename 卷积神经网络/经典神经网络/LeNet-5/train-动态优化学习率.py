import json
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Resize
from utils.train_utils import DatasetSpliter
from LeNet5 import LeNet5


def save(_model, _loss, _epoch, _lr):
    state_dict = _model.state_dict()
    torch.save(state_dict, f'./weights/LeNet5_{_epoch}.pt')
    with open('./weights/meta.json', 'w') as file:
        json.dump({
            'loss': _loss,
            'epoch': _epoch,
            'lr': _lr
        }, file)


def load():
    _model = LeNet5()
    _loss = float('inf')
    _epoch = 0
    _lr = 1e-3

    try:
        with open('./weights/meta.json', 'r') as file:
            meta = json.load(file)
        _loss = meta['loss']
        _epoch = meta['epoch']
        _lr = meta['lr']
        state_dict = torch.load(f'./weights/LeNet5_{_epoch}.pt', weights_only=True)
        _model.load_state_dict(state_dict)
        print('模型加载成功')
    except:
        print('模型加载失败')

    return _model, _loss, _epoch, _lr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 1
batch_size = 100

transform = Compose([
    ToTensor(),
    Resize((32, 32), antialias=True)
])
ds = MNIST(root='./data', train=True, transform=transform, download=True)
ds_spliter = DatasetSpliter(ds, batch_size)

model, best_loss, total_epoch, lr = load()
model.to(device)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 学习率调度器，用于自动优化学习率
# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,  # 存储和使用学习率的优化器
    mode='min',  # 最大优化还是最小优化，监视学习率下降时，则是最小优化问题，输入 min，否则输入 max
    factor=0.1,  # 每次优化后学习率下降的倍数
    patience=3,  # 在多少个epoch内，如果优化器没有优化，则学习率下降
    threshold=0.0001,  # 优化器优化的阈值，当损失值变化量低于阈值，则认为学习率需要调整了
    cooldown=1,  # 学习率下降后，经过多少个epoch，再重新监视学习率下降
    min_lr=1e-6  # 学习率的最小值
)


t_total_loss = 0.
t_count = 0


def train():
    global t_total_loss, t_count
    print('train start')
    model.train()
    for i, (inputs, labels) in enumerate(train_dl):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y = model(inputs)
        loss = loss_fn(y, labels)
        t_total_loss += loss.item()
        t_count += 1
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'batch: {i + 1}; avg loss: {t_total_loss / t_count}')
    print(f'train over; avg loss: {t_total_loss / t_count}')


v_total_loss = 0.
v_count = 0


def val():
    global v_total_loss, v_count
    print('val start')
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            y = model(inputs)
            loss = loss_fn(y, labels)
            v_total_loss += loss.item()
            v_count += 1
    avg_loss = v_total_loss / v_count
    print(f'val over; {avg_loss}')


for epoch in range(EPOCH):
    print(f'EPOCH: [{epoch + 1}/{EPOCH}]')
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    train()
    val()
    avg_loss = v_total_loss / v_count
    scheduler.step(avg_loss)
    total_epoch += 1

print('训练结束')
avg_loss = v_total_loss / v_count
if avg_loss < best_loss:
    save(model, avg_loss, total_epoch, optimizer.param_groups[0]['lr'])
    print('保存模型')
else:
    print('未保存模型')
