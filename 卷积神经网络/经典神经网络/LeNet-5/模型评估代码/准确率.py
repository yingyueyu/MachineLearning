# 准确率 = 正确预测的样本 / 所有样本
import random
import sys  # 导入系统包

sys.path.append('../')  # 追加路径到环境变量中，被追加的路径下的文件就可以直接引用

import torch
from LeNet5 import LeNet5
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader

# 导入模型
# map_location='cpu': 转换模型格式到对应设备上
# weights_only=True: 添加模型的可信任度，告诉程序这是个安全的可信任的模型
state_dict = torch.load('../weights/LeNet5_500.pt', map_location='cpu', weights_only=True)
model = LeNet5()
model.load_state_dict(state_dict)
model.eval()  # 开启评估模式

transform = Compose([
    Grayscale(),
    ToTensor(),
    Resize((32, 32), antialias=True)
])

# 加载验证集
ds = MNIST(root='../data', train=True, transform=transform)
ds_len = len(ds)
# 将 20% 的数据作为验证数据
val_len = int(ds_len * 0.2)
# 随机索引
# 为了每次随机结果相同，需要固定随机种子
random.seed(100)
# 第一个参数: 采样范围
# 第二个参数: 采样个数
indices = random.sample(range(ds_len), val_len)
# 创建子集
val_ds = Subset(ds, indices)
dl = DataLoader(val_ds, batch_size=100, shuffle=False)

# 总共正确的个数
true_total_count = 0

with torch.inference_mode():
    for i, (inputs, labels) in enumerate(dl):
        # 预测
        y = model(inputs)
        values, indices = torch.topk(y, k=1, dim=1)
        # 将维度为 100x1 的 indices 转换成 100
        indices = torch.squeeze(indices, 1)
        # 对比
        mask = labels == indices
        # 求和 计算正确的个数
        true_count = mask.sum()
        true_total_count += true_count

print(f'准确率: {true_total_count / val_len * 100:.2f}%')
