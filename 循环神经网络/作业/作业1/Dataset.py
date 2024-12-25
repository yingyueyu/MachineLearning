import torch
from torch.utils.data import Dataset


class TempDataset(Dataset):
    def __init__(self, C=5):
        super().__init__()
        n1 = torch.randint(10, 16, (5,))
        n2 = torch.randint(15, 21, (5,))
        n3 = torch.randint(18, 25, (5,))
        n4 = torch.randint(15, 23, (5,))
        self.n = torch.cat([n1, n2, n3, n4], dim=0).to(torch.float)
        self.inputs = []
        self.labels = []
        # 计算迭代次数
        iter_count = len(self.n) - C
        for i in range(iter_count):
            # 获取输入
            inp = self.n[i:i + C]
            # 获取标签
            label = self.n[i + C]
            self.inputs.append(inp)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        label = self.labels[idx]
        return inp, label


if __name__ == '__main__':
    ds = TempDataset()
    print(len(ds))
    print(ds[0])
