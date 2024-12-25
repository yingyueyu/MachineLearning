import copy
import json

import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor

from 数字和英文识别模型 import TextRecognition


def save(_model, _best_loss):
    state_dict = _model.state_dict()
    torch.save(state_dict, 'weights/without_peft_model.pt')
    with open('weights/without_peft_train_meta.json', 'w') as file:
        json.dump({'best_loss': _best_loss}, file)


def load(device):
    _model = TextRecognition(device)
    _best_loss = float('inf')
    try:
        state_dict = torch.load('weights/without_peft_model.pt')
        _model.load_state_dict(state_dict)
        with open('weights/without_peft_train_meta.json', 'r') as file:
            _best_loss = json.load(file)['best_loss']
        print('加载模型成功')
    except:
        print('未找到预训练模型，初始化空模型')
    return _model, _best_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 1
lr = 1e-3
batch_size = 100

# 手动下载地址 https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
# 英文字母的label是从1开始的数字，1代表A
# ds = EMNIST(root='./data', split='letters', train=True, download=False)
ds = EMNIST(root='./data', split='letters', train=True, download=False, transform=ToTensor())
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

model, best_loss = load(device)
# 备份一个原始模型
model_copy = copy.deepcopy(model)

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
        optimizer.zero_grad()
        inputs = inputs.to(device)
        # 原始数据集 labels 取值范围是 1~26
        # 这里平移 10 位，并取索引值，所以英文字母的label的范围是 10~35
        labels = (labels + 10 - 1).to(device)
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

# 训练完后对比训练参数
print(f'======================={"检查更新了哪些参数":^20}==========================')

src_param_dict = dict(model_copy.named_parameters())

# 遍历所有参数
for name, value in model.named_parameters():
    param_before = src_param_dict[name]
    # 比较微调后模型和原始模型，是否近似相等
    if torch.allclose(value, param_before):
        # 近似相当的当作没有更新的参数
        print(f"Parameter {name:<15} | {param_before.numel():>8} parameters | not updated")
    else:
        # 否则当作被更新的参数
        print(f"Parameter {name:<15} | {value.numel():>8} parameters | updated")

# avg_loss = total_loss / count
#
# if avg_loss < best_loss:
#     save(model, avg_loss)
