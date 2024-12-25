import time

import torch
import torch.nn.functional as F

from 声明回归模型 import ChatBot

# 加载模型
model = ChatBot()
state_dict = torch.load('weights/model.pt', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

text = 'hey how are you'
text = list(set(text))
text = sorted(text)
print(text)


# 转换器
def transform(inp):
    idx = [text.index(c) for c in inp]
    t = F.one_hot(torch.tensor(idx), len(text))
    t = t.sum(dim=0)
    t[t > 1] = 1
    t = t.to(torch.float)
    return t


inp = 'hey h'

print(f'输入: {inp}')
print(f'输出: {inp}', end='')

# 在推理模式上下文下让模型预测结果
with torch.inference_mode():
    for i in range(10):
        # 转换
        x = transform(inp)
        y = model(x)
        # 获取预测概率最大的索引
        idx = y.argmax(dim=0).item()
        # 获取索引对应的文本
        txt = text[idx]
        time.sleep(1)
        print(txt, end='')
        # 构造下一轮循环的输入
        inp = inp[1:] + txt
