import torch
import torchaudio.transforms as T

from model import w2v2
from dataset import audio_tensor, valid_lengths, max_len, vocab

w2v2.load_state_dict(torch.load('../model.pt'))
w2v2.eval()

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


def audio_check(idx):
    emission, _ = w2v2(data[idx].unsqueeze(0), None)
    print(emission.shape)

    idx = emission.softmax(-1).argmax(-1)
    print(idx)
    tokens = [vocab[i] for i in idx[0].numpy()]
    print(tokens)
    return tokens


map = torch.zeros(9, 9, dtype=torch.int)
pos = [4, 4]
map[pos[0], pos[1]] = 1

while True:
    print(map)
    dir = input('请输入指令: ')
    tokens = None
    if dir == 'up':
        tokens = audio_check(0)
    elif dir == 'down':
        tokens = audio_check(1)
    elif dir == 'left':
        tokens = audio_check(2)
    elif dir == 'right':
        tokens = audio_check(3)

    map[pos[0], pos[1]] = 0
    if '上' in tokens:
        pos[0] -= 1
    elif '下' in tokens:
        pos[0] += 1
    elif '左' in tokens:
        pos[1] -= 1
    elif '右' in tokens:
        pos[1] += 1

    map[pos[0], pos[1]] = 1
