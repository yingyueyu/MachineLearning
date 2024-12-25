import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class TextDataset(Dataset):
    # 采集数据的窗口大小
    def __init__(self, C=5, transform=None, target_transform=None):
        super().__init__()
        self.inputs = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        # 原始文本
        text = 'hey how are you'
        # 窗口滑动的次数
        iter_count = len(text) - C
        # 生成数据集
        for i in range(iter_count):
            self.inputs.append(text[i: i + C])
            self.labels.append(text[i + C])

    def __len__(self):
        # 用标签数量作为数据集长度
        return len(self.labels)

    def __getitem__(self, index):
        inp = self.inputs[index]
        label = self.labels[index]
        if self.transform is not None:
            inp = self.transform(inp)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # 将字符串转换为数字
        return inp, label


if __name__ == '__main__':
    text = 'hey how are you'
    # 字符去重
    text = list(set(text))
    # 按 ASCii 码排序
    text = sorted(text)
    print(text)


    # 输入转换器
    def transform(inp):
        # text.index(c): 查询字符 c 在列表 text 中的索引
        idx = [text.index(c) for c in inp]

        # 二元词袋编码: 在独热编码基础上，出现那些字符就设置为 1，没出现的字符就是 0

        # 独热编码
        # 参数一: 索引
        # 参数二: 独热编码的长度（总共有多少字）
        t = F.one_hot(torch.tensor(idx), len(text))
        t = t.sum(dim=0)
        # 限制编码大小
        t[t > 1] = 1
        return t


    # 标签转换器
    def target_transform(label):
        return text.index(label)


    ds = TextDataset(transform=transform, target_transform=target_transform)
    print(len(ds))
    inp, label = ds[0]
    print(inp)
    print(label)
