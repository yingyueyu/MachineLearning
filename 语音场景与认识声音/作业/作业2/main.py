import torch
import torchaudio.transforms as T

from model import w2v2
from dataset import audio_tensor, vocab

w2v2.load_state_dict(torch.load('model.pt'))
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
data = data.reshape(4, -1)


# 识别声音
# idx: 哪一段声音
def audio_check(idx):
    with torch.inference_mode():
        emission, _ = w2v2(data[idx].unsqueeze(0), None)
    # 求概率分布
    tokens = emission.softmax(-1).argmax(-1)
    return [vocab[token] for token in tokens[0].numpy()]


# 地图
map = torch.zeros(9, 9, dtype=torch.int)
# 1 的坐标
pos = [4, 4]

while True:
    map[pos[0], pos[1]] = 1
    print(map)
    op = input('请输入操作: ')
    # 要包放的音频索引
    idx = None
    if op == 'up':
        idx = 0
    elif op == 'down':
        idx = 1
    elif op == 'left':
        idx = 2
    elif op == 'right':
        idx = 3

    # 识别声音
    tokens = audio_check(idx)

    # 还原 0
    map[pos[0], pos[1]] = 0

    if '上' in tokens:
        pos[0] -= 1
    elif '下' in tokens:
        pos[0] += 1
    elif '左' in tokens:
        pos[1] -= 1
    elif '右' in tokens:
        pos[1] += 1
