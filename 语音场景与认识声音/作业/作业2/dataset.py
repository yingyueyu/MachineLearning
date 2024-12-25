import torch
import torchaudio

direction = 'up down left right'.split()

audio_data = [torchaudio.load(f'audios/{d}.wav')[0][0] for d in direction]

# 统计音频信号的有效长度
# 音频有效长度
valid_lengths = []
# 音频有效长度中的最大值
max_len = 0

valid_audio_data = []

for ad in audio_data:
    # 获取不为零的数据索引
    idx = torch.nonzero(ad).reshape(-1)
    # 获取有效音频张量
    valid_tensor = ad[idx[0]:idx[-1] + 1]
    valid_audio_data.append(valid_tensor)
    # 保存音频的有效长度
    valid_len = valid_tensor.shape[0]
    valid_lengths.append(valid_len)
    if valid_len > max_len:
        max_len = valid_len

# 所有音频的张量
audio_tensor = torch.zeros(len(valid_audio_data), max_len)
print(audio_tensor.shape)

for i in range(len(valid_audio_data)):
    # 有效音频数据
    ad = valid_audio_data[i]
    # 赋值 ad 到有效长度 valid_lengths[i] 的位置
    audio_tensor[i][:valid_lengths[i]] = ad

print(audio_tensor)

vocab = '- * 上 下 左 右'.split()
dictionary = {word: i for i, word in enumerate(vocab)}
# 音频对应文本
label = [[dictionary[w] for w in '* 上'.split()], [dictionary[w] for w in '* 下'.split()],
         [dictionary[w] for w in '* 左'.split()], [dictionary[w] for w in '* 右'.split()]]
print(label)
# 文本长度
target_lengths = torch.tensor([len(l) for l in label])

tokens = torch.tensor(label)
