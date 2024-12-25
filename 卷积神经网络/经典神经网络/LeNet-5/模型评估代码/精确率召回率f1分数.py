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

# 为 10 个数字创建统计存储的集合
# TPCount: 所有正例 3
# val_len - TPCount: 所有负例
# 例如:
# 样本为 3，模型预测为 3，则应该在 i == 3 的字典中，TP += 1
# 样本为 3，模型预测为 5，则应该在 i == 5 的字典中，FP += 1
stats = {i: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'TPCount': 0} for i in range(10)}

with torch.inference_mode():
    for i, (inputs, labels) in enumerate(dl):
        # 模型预测
        y = model(inputs)

        for j in range(y.shape[0]):
            idx = torch.argmax(y[j]).item()
            label = labels[j].item()
            if idx == label:
                # 预测正确
                # 统计真阳性
                stats[label]['TP'] += 1
            else:
                # 预测错误，发生误报
                # 统计假阳性
                stats[idx]['FP'] += 1
            # 统计正例个数
            stats[label]['TPCount'] += 1

# 计算 TN FN
for key in stats.keys():
    # 计算所有负例个数
    FPCount = val_len - stats[key]['TPCount']
    stats[key]['TN'] = FPCount - stats[key]['FP']
    stats[key]['FN'] = stats[key]['TPCount'] - stats[key]['TP']

print(stats)


# 宏平均
precision_macro = 0.
recall_macro = 0.
f1_macro = 0.
# 加权平均
precision_weighted = 0.
recall_weighted = 0.
f1_weighted = 0.

# 统计精确率、召回率、F1分数
for key in stats.keys():
    # 精确率
    stats[key]['Precision'] = stats[key]['TP'] / (stats[key]['TP'] + stats[key]['FP'])
    # 召回率
    stats[key]['Recall'] = stats[key]['TP'] / (stats[key]['TP'] + stats[key]['FN'])
    # F1分数
    stats[key]['F1'] = 2 * stats[key]['Precision'] * stats[key]['Recall'] / (
                stats[key]['Precision'] + stats[key]['Recall'])

    # 统计宏平均的分子
    precision_macro += stats[key]['Precision']
    recall_macro += stats[key]['Recall']
    f1_macro += stats[key]['F1']
    # 统计加权平均的分子
    precision_weighted += stats[key]['Precision'] * stats[key]['TPCount']
    recall_weighted += stats[key]['Recall'] * stats[key]['TPCount']
    f1_weighted += stats[key]['F1'] * stats[key]['TPCount']

print(stats)

precision_macro /= 10
recall_macro /= 10
f1_macro /= 10

precision_weighted /= val_len
recall_weighted /= val_len
f1_weighted /= val_len

print(f'精确率 宏平均: {precision_macro}')
print(f'召回率 宏平均: {recall_macro}')
print(f'F1分数 宏平均: {f1_macro}')

print(f'精确率 加权平均: {precision_weighted}')
print(f'召回率 加权平均: {recall_weighted}')
print(f'F1分数 加权平均: {f1_weighted}')
