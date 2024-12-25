import torch
import random

from Dataset import TempDataset
from TempModel import TempModel

torch.manual_seed(100)

# 加载数据
ds = TempDataset()

# 加载模型
state_dict = torch.load('model.30000.pt', weights_only=True)
model = TempModel(10)
model.load_state_dict(state_dict)
model.eval()

# 随机样本
C = 5
idx = random.randint(0, 14)
print(idx)
x = ds.n[idx: idx + C]
print(x)

# 预测结果
y, h = model(x.view(-1, 1))
print(ds.n)
print(y[-1])
