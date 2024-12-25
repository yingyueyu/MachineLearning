import torch

from models import LotteryModel
from dataset import LotteryDataset

# 加载模型
model = LotteryModel()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True, map_location='cpu'))
model.eval()

# 加载数据集
ds = LotteryDataset()

with torch.inference_mode():
    for i in range(len(ds)):
        inputs, labels = ds[i]
        y, h, c = model(inputs.unsqueeze(0))
        y = y.argmax(-1)
        print(torch.equal(y[0], labels))
