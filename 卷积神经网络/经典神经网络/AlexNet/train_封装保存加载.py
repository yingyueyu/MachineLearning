import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize

from utils.train_utils import DatasetSpliter
from 水果数据集 import FruitDataset
from AlexNet import AlexNet
from utils.train_utils import save, load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 声明超参数
EPOCH = 1
batch_size = 100

# 创建数据集
transform = Compose([
    ToTensor(),
    Resize((224, 224), antialias=True)
])

fruit_dict = {
    'Banana': 0,
    'Corn': 1,
    'Watermelon': 2
}

ds = FruitDataset(transform, lambda label: fruit_dict[label])
ds_spliter = DatasetSpliter(ds, batch_size)

# 创建模型
model, best_loss, total_epoch, lr, meta = load('./weights', AlexNet, {'num_classes': 3})
model.to(device)

# 创建损失函数、优化器、学习率调度器
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=1e-4,
    cooldown=1,
    min_lr=1e-6
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

        if (i + 1) % 5 == 0:
            print(f'batch: {i + 1}; loss: {t_total_loss / t_count}')
    print(f'train over; loss: {t_total_loss / t_count}')


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
    print(f'val over; loss {v_total_loss / v_count}')


# 训练循环
for epoch in range(EPOCH):
    print(f'EPOCH: [{epoch + 1}/{EPOCH}]')
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    train()
    val()
    # 调用调度器，调整学习率
    scheduler.step(v_total_loss / v_count)
    total_epoch += 1

avg_loss = v_total_loss / v_count
print(f'训练结束; avg_loss: {avg_loss}')
if avg_loss < best_loss:
    save('./weights', model, avg_loss, total_epoch, optimizer.param_groups[0]['lr'])
    print('模型已保存')
else:
    print('模型未保存')
