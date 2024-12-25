import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from 数字识别模型 import NumberRecognition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

model = NumberRecognition()
model.load_state_dict(torch.load('weights/model.pt', map_location=device))
model.to(device).eval()

# 测试集
ds = MNIST(root='./data', train=False, download=False, transform=ToTensor())
dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

# 正确的个数
correct_count = 0
# 总个数
count = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dl):
        y = model(inputs)
        # 和真实数据对比
        r = torch.argmax(y, dim=-1) == labels
        # 添加样本数
        count += len(labels)
        # 将为真的样本相加，得到正确的个数
        correct_count += r.sum().item()

print(f'正确率: {round(correct_count / count * 100, 2)}%')
