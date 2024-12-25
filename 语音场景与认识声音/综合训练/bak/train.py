import torch.optim
from torch import nn

from dataset import audio_data, tokens
from model import w2v2

# 超参数
EPOCH = 100
lr = 1e-2

# 优化器
optimizer = torch.optim.Adam(w2v2.parameters(), lr=lr)
# 损失函数 https://pytorch.org/docs/2.3/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss
loss_fn = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

# 处理tokens，将他转换成带批次的张量数据
tokens = torch.tensor(tokens)

# 定义损失函数的参数
# 输入长度为 1 批次 314 帧
input_lengths = torch.tensor([3], dtype=torch.int)

# 输出长度为 tokens 总长度，且 tokens 中不能包含 0，0 代表 blank
target_lengths = torch.tensor([len(tokens)])

# 展平或变形，让音频输入的形状（格式）变成 waveform 类似的格式
# 形状为 (N, T)
audio_data = audio_data.reshape(1, -1)

# 开启训练模式
w2v2.train()
for epoch in range(EPOCH):
    optimizer.zero_grad()
    # 模型推理，前向传播
    emission, _ = w2v2(audio_data, None)
    emission = emission.log_softmax(2).transpose(0, 1)
    input_lengths = torch.tensor([emission.shape[0]])
    # 求损失
    loss = loss_fn(emission, tokens, input_lengths=input_lengths, target_lengths=target_lengths)
    print(loss.item())
    # 反向传播
    loss.backward()
    # 优化
    optimizer.step()
