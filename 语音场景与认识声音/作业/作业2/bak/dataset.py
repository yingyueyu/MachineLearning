import torch
import torchaudio

vocab = '- * 上 下 左 右'.split()
dictionary = {word: i for i, word in enumerate(vocab)}
tokens = [[dictionary[word] for word in '* 上'.split()], [dictionary[word] for word in '* 下'.split()],
          [dictionary[word] for word in '* 左'.split()], [dictionary[word] for word in '* 右'.split()]]

target_lengths = torch.tensor([len(token) for token in tokens])

direction = 'up down left right'.split()

datas = [torchaudio.load(f'../audios/{d}.wav')[0][0] for d in direction]
sample_rate = 16000
print(datas)
max_len = 0
# 有效长度
valid_lengths = []
for d in datas:
    # 非零值索引
    nonzero_idx = torch.nonzero(d).reshape(-1)
    # 最后一个索引 + 1 则为有效长度
    valid_len = nonzero_idx[-1] + 1
    valid_lengths.append(valid_len)
    if valid_len > max_len:
        max_len = valid_len
print(max_len)
print(valid_lengths)

# 输入张量
audio_tensors = torch.zeros(len(datas), max_len)

for i, d in enumerate(datas):
    audio_tensors[i][:valid_lengths[i]] = d[:valid_lengths[i]]

print(audio_tensors)
