import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt

waveform, sample_rate = torchaudio.load('../../audios/376967__kathid__goodbye-high-quality.mp3')
print(waveform.shape)
print(sample_rate)

# 合并两通道的音频
waveform = waveform.sum(dim=0)

# 构造横坐标 t 时间，单位 秒
time = len(waveform) / sample_rate
t = np.linspace(0, time, len(waveform))

# 振幅
a = torch.abs(torch.fft.fft(waveform))
# 频率
f = torch.fft.fftfreq(len(waveform), 1 / sample_rate)

a = a[:len(a) // 2]
f = f[:len(f) // 2]

fig, ax = plt.subplots(2)

# 时域图
ax[0].plot(t, waveform.numpy())
# 频域图
ax[1].plot(f.numpy(), a.numpy())

plt.show()
