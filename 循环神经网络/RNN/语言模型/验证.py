import time

import torch
import torch.nn as nn
import torchtext

# 关闭警告
torchtext.disable_torchtext_deprecation_warning()

from 循环神经网络.RNN.语言模型.语言模型 import LangModel
from 数据集与分词器 import Tokenizer

# 加载模型
state_dict = torch.load('weights/model.pt', weights_only=True, map_location='cpu')
model = LangModel()
model.load_state_dict(state_dict)
model.eval()

# 加载分词器
tokenizer = Tokenizer()

input_txt = '<sos> how are you?'
tokens = tokenizer.encode(input_txt)

print(f'输入: {input_txt}')
print(f'输出: {input_txt}', end='')

# 最大循环次数
max_iter = 10
# 跳出循环的值，当模型预测出这个值，则跳出循环
break_value = '<eos>'

for i in range(max_iter):
    inputs = nn.functional.one_hot(torch.tensor(tokens), len(tokenizer)).to(torch.float)
    inputs = inputs.unsqueeze(0)
    # 推理模式
    with torch.inference_mode():
        y, h = model(inputs)
    idx = torch.argmax(y, dim=-1)
    # print(idx)
    # 解码
    tokens = tokenizer.decode(idx[0].numpy())
    # print(tokens)
    # 打印最后一个值
    time.sleep(1)
    print(f' {tokens[-1]}', end='')

    # 判断是否跳出循环
    if tokens[-1] == break_value:
        break

    # 编码
    tokens = tokenizer.encode(' '.join(tokens))
