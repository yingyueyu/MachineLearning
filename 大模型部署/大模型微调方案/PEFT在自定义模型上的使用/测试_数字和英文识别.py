import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import ToTensor

from 数字和英文识别模型 import TextRecognition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

model = TextRecognition(device)
model.load_state_dict(torch.load('weights/without_peft_model.pt', map_location=device))
model.to(device).eval()

# 测试集
ds1 = MNIST(root='./data', train=False, download=False, transform=ToTensor())
dl1 = DataLoader(ds1, batch_size=batch_size, shuffle=False)

ds2 = EMNIST(root='./data', split='letters', train=False, download=False, transform=ToTensor())
dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=False)

# 正确的个数
correct_count = 0
# 总个数
count = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dl1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        y = model(inputs)
        # 和真实数据对比
        r = torch.argmax(y, dim=-1) == labels
        # 添加样本数
        count += len(labels)
        # 将为真的样本相加，得到正确的个数
        correct_count += r.sum().item()

    for i, (inputs, labels) in enumerate(dl2):
        inputs = inputs.to(device)
        labels = (labels + 10 - 1).to(device)
        y = model(inputs)
        # 和真实数据对比
        r = torch.argmax(y, dim=-1) == labels
        # 添加样本数
        count += len(labels)
        # 将为真的样本相加，得到正确的个数
        correct_count += r.sum().item()

print(f'正确率: {round(correct_count / count * 100, 2)}%')
