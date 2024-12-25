import torch.optim
import torchaudio.transforms as T
from torch import nn

from dataset import sample_rate, audio_tensors, tokens, valid_lengths, target_lengths
from model import w2v2

# 音频提取特征
# 梅尔频率倒谱系数 转换器
transform = T.MFCC(
    sample_rate=sample_rate,
    # 变换器输出的个数，类似卷积的输出通道数
    # n_mfcc 不能大于 n_mels
    n_mfcc=23,
    # n_mels: 梅尔滤波器的个数
    melkwargs={'n_fft': 400, 'win_length': 256, 'hop_length': 256, 'n_mels': 23, 'power': 2}
)

optim = torch.optim.Adam(w2v2.parameters(), lr=0.001)
loss_fn = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

w2v2.train()
for epoch in range(100):
    optim.zero_grad()
    emission, vl = w2v2(audio_tensors, torch.tensor(valid_lengths))
    emission = emission.log_softmax(-1).transpose(0, 1)
    loss = loss_fn(emission, torch.tensor(tokens), input_lengths=vl, target_lengths=target_lengths)
