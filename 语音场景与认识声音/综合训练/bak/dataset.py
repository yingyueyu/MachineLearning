import torchaudio
import torchaudio.transforms as T

waveform, sample_rate = torchaudio.load('../独坐敬亭山.wav')

transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=13,
    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
)

audio_data = transform(waveform)

# 词库
vocab = '- 众 鸟 高 飞 尽 孤 云 独 去 闲 相 看 两 不 厌 只 有 敬 亭 山'.split(' ')

dictionary = {word: i for i, word in enumerate(vocab)}

# 标签: 音频中对应的文本
label = '众 鸟 高 飞 尽 孤 云 独 去 闲 相 看 两 不 厌 只 有 敬 亭 山'.split(' ')

# 将标签转换成token
tokens = [dictionary[word] for word in label]

print(tokens)
