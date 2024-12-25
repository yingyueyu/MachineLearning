import torch

from 模型 import LangModel
from 数据集 import Tokenizer

# 加载模型
model = LangModel()
state_dict = torch.load('weights/L.pt', weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# 构造一组输入
x = torch.tensor([0, 1, 2], dtype=torch.float)
x = x.view(3, 1)
x = x.expand(3, 3)
x = x.chunk(3, dim=0)
x = torch.cat(x, dim=-1)
x = x.T

with torch.inference_mode():
    y, h = model(x)

idx = y.argmax(dim=-1)

print(idx)

tokenizer = Tokenizer()

tokens = [tokenizer.decode(idx[i].numpy()) for i in range(idx.shape[0])]

print(tokens)
