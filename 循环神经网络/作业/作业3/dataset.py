import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class LotteryDataset(Dataset):
    def __init__(self, C=5):
        super().__init__()

        random.seed(100)

        # 生成数据
        self.data = [random.sample(range(1, 17), 6) + random.sample(range(1, 9), 2) for i in range(20)]

        self.inputs = []
        self.labels = []

        iter_count = len(self.data) - C

        for i in range(iter_count):
            self.inputs.append(self.data[i:i + C])
            self.labels.append(self.data[i + 1:i + C + 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = torch.tensor(self.inputs[idx])
        # 独热编码
        inp = torch.stack([F.one_hot(inp[i], 17) for i in range(inp.shape[0])]).to(torch.float)
        label = torch.tensor(self.labels[idx])
        return inp, label


if __name__ == '__main__':
    ds = LotteryDataset()
    print(ds[0])
