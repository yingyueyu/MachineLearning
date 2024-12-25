import random
import sys  # 导入系统包

sys.path.append('../../经典神经网络/LeNet-5')  # 追加路径到环境变量中，被追加的路径下的文件就可以直接引用

import torch
from LeNet5 import LeNet5
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader

# 导入模型
# map_location='cpu': 转换模型格式到对应设备上
# weights_only=True: 添加模型的可信任度，告诉程序这是个安全的可信任的模型
state_dict = torch.load('../../经典神经网络/LeNet-5/weights/LeNet5_500.pt', map_location='cpu', weights_only=True)
model = LeNet5()
model.load_state_dict(state_dict)
model.eval()  # 开启评估模式

transform = Compose([
    Grayscale(),
    ToTensor(),
    Resize((32, 32), antialias=True)
])

# 加载验证集
ds = MNIST(root='../../经典神经网络/LeNet-5/data', train=True, transform=transform)
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

# 为 10 个数字创建统计存储的集合
# TPCount: 所有正例 3
# val_len - TPCount: 所有负例
# 例如:
# 样本为 3，模型预测为 3，则应该在 i == 3 的字典中，TP += 1
# 样本为 3，模型预测为 5，则应该在 i == 5 的字典中，FP += 1
# {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'TPCount': 0}
# 0: TP
# 1: FP
# 2: TN
# 3: FN
# 4: TPCount
# 5: TNCount
# 6: Precision
# 7: Recall
# 8: F1
stats = torch.zeros(10, 9, dtype=torch.int)

with torch.inference_mode():
    for i, (inputs, labels) in enumerate(dl):
        y = model(inputs)
        # 100x10
        # top1
        values, indices = torch.topk(y, 1, dim=1)
        indices = indices.squeeze(1)
        # 判断正确的bool张量
        mask = indices == labels
        # 找出正确的数字有哪些
        # 提取出正确的数字
        true_value = labels[mask]
        # 统计正确数字的个数
        TP = torch.bincount(true_value, minlength=10)
        # 统计 TP
        stats[:, 0] += TP
        # 提取计算错的数据
        false_value = indices[~mask]
        # 统计错误数字的个数
        FP = torch.bincount(false_value, minlength=10)
        stats[:, 1] += FP
        # 所有正样本的个数
        TPCount = torch.bincount(labels, minlength=10)
        stats[:, 4] += TPCount

# 统计 TNCount
stats[:, 5] = val_len - stats[:, 4]

# 统计 TN
stats[:, 2] = stats[:, 5] - stats[:, 1]

# 统计 FN
stats[:, 3] = stats[:, 4] - stats[:, 0]

print(stats)

stats = stats.to(torch.float)

# 精确率
stats[:, 6] = stats[:, 0] / (stats[:, 0] + stats[:, 1])
# 召回率
stats[:, 7] = stats[:, 0] / (stats[:, 0] + stats[:, 3])
# F1
stats[:, 8] = 2 * stats[:, 6] * stats[:, 7] / (stats[:, 6] + stats[:, 7])

print(stats)

# 宏平均
precision_macro = stats[:, 6].mean()
recall_macro = stats[:, 7].mean()
f1_macro = stats[:, 8].mean()

# 加权平均
precision_weighted = (stats[:, 6] * stats[:, 4]).sum() / val_len
recall_weighted = (stats[:, 7] * stats[:, 4]).sum() / val_len
f1_weighted = (stats[:, 8] * stats[:, 4]).sum() / val_len

print(f'精确率 宏平均: {precision_macro * 100:.6f}%')
print(f'召回率 宏平均: {recall_macro * 100:.6f}%')
print(f'F1分数 宏平均: {f1_macro * 100:.6f}%')

print(f'精确率 加权平均: {precision_weighted * 100:.6f}%')
print(f'召回率 加权平均: {recall_weighted * 100:.6f}%')
print(f'F1分数 加权平均: {f1_weighted * 100:.6f}%')
