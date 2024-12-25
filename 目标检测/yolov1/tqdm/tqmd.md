# tqdm



## 安装

```
pip install tqdm
```



## 使用



```
import time
from tqdm import *
for i in tqdm(range(1000)):
    time.sleep(.01)   #进度条每0.01s前进一次，总时间为1000*0.01=10s 

# 运行结果如下
100%|██████████| 1000/1000 [00:10<00:00, 93.21it/s]  

```

上述使用还可以使用`trange`

`trange(i) `是 `tqdm(range(i)) `的简单写法

```
from tqdm import trange

for i in trange(1000):
    time.sleep(.01)

# 运行结果如下
100%|██████████| 1000/1000 [00:10<00:00, 93.21it/s]  
```



设置进度条描述

```
import time
from tqdm import tqdm

pbar = tqdm(["a","b","c","d"])

for char in pbar:
    pbar.set_description("Processing %s" % char) # 设置描述
    time.sleep(1)  # 每个任务分配1s
    
# 结果如下
  0%|          | 0/4 [00:00<?, ?it/s]

Processing a:   0%|          | 0/4 [00:00<?, ?it/s]

Processing a:  25%|██▌       | 1/4 [00:01<00:03,  1.01s/it]

Processing b:  25%|██▌       | 1/4 [00:01<00:03,  1.01s/it]

Processing b:  50%|█████     | 2/4 [00:02<00:02,  1.01s/it]

Processing c:  50%|█████     | 2/4 [00:02<00:02,  1.01s/it]

Processing c:  75%|███████▌  | 3/4 [00:03<00:01,  1.01s/it]

Processing d:  75%|███████▌  | 3/4 [00:03<00:01,  1.01s/it]

Processing d: 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]

```



## 深度学习中的使用

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

# Set device
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(device)
# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train Network

for epoch in range(num_epochs):
    # for data,targets in tqdm(train_loadr,leave=False) # 进度显示在一行
    for data,targets in tqdm(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores,targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gardient descent or adam step
        optimizer.step()

```

对应5个eopch的5个进度显示
如果我们想要它显示在一行，在tqdm中添加`leave=False`参数即可

```
for data,targets in tqdm(train_loadr,leave=False) # 进度显示在一行
```



我们将`tqdm`加到`train_loader`无法得到索引，要如何得到索引呢？可以使用下面的代码

```
for index,（data,targets） in tqdm(enumerate(train_loader),total=len(train_loader),leave = True):
```



我们觉得还有点不太满足现在的进度条，我们得给他加上我们需要的信息，比如准确率，loss值，如何加呢？

```python
for epoch in range(num_epochs):
    losses = []
    accuracy = []
    # for data,targets in tqdm(train_loadr,leave=False) # 进度显示在一行
    loop = tqdm((train_loader), total = len(train_loader))
    for data,targets in loop:
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        _,predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])
        accuracy.append(running_train_acc)
        # gardient descent or adam step
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss = loss.item(),acc = running_train_acc)

```

