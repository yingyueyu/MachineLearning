import random

import torch
import torch.nn.functional as F

from 循环神经网络.作业.作业4.models import CardModel_GRU

# 扑克牌
cards = [[i, j] for j in range(4) for i in range(2, 15)]
print(len(cards))

# 洗牌
random.shuffle(cards)

print(cards)

# 抽牌
c1 = cards.pop(0)
c2 = cards.pop(0)
print(c1)
print(c2)
print(len(cards))

inp = torch.tensor([c1, c2])

# 编码
# 对数字部分进行编码
num = F.one_hot(inp[:, 0], 15)
# 对花色进行编码
face = F.one_hot(inp[:, 1], 4)
# 连接数字和花色
inp = torch.cat([num, face], dim=1)

print(inp)

# 创建模型
model = CardModel_GRU()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True, map_location='cpu'))
model.eval()

with torch.inference_mode():
    y, h = model(inp.unsqueeze(1))
print(y)
