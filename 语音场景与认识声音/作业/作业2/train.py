import torch
import torchaudio.transforms as T

from torch import nn

from dataset import audio_tensor, tokens, valid_lengths, max_len, target_lengths
from model import w2v2

EPOCH = 100
lr = 1e-3

optim = torch.optim.Adam(w2v2.parameters(), lr=lr)
loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# 音频提取特征
# 梅尔频率倒谱系数 转换器
transform = T.MFCC(
    sample_rate=16000,
    # 变换器输出的个数，类似卷积的输出通道数
    # n_mfcc 不能大于 n_mels
    n_mfcc=23,
    # n_mels: 梅尔滤波器的个数
    melkwargs={'n_fft': 400, 'win_length': 256, 'hop_length': 256, 'n_mels': 23, 'power': 2}
)

data = transform(audio_tensor)
data = data.reshape(data.shape[0], -1)
# 最大长度的缩放比例
rate = max_len / data.shape[1]
# 缩放后的有效长度
valid_lens = [l // rate + 1 for l in valid_lengths]

total_loss = 0.
count = 0

w2v2.load_state_dict(torch.load('model.pt'))
w2v2.train()
for epoch in range(EPOCH):
    optim.zero_grad()
    emission, vl = w2v2(data, torch.tensor(valid_lens, dtype=torch.int))
    emission = emission.log_softmax(-1).transpose(0, 1)
    loss = loss_fn(emission, tokens, input_lengths=vl, target_lengths=target_lengths)
    total_loss += loss.item()
    count += 1
    loss.backward()
    optim.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(w2v2.state_dict(), 'model.pt')
