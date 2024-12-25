import torch
from torch.utils.data import Dataset
import pandas as pd


class WikiDataset(Dataset):
    # split: 获取哪个数据集，train 训练集 test 测试集 validation 验证集
    # mode: 数据集模式，cbow 代表连续词袋，skip-gram 代表跳元模型
    # C: 每个词的上下文窗口大小
    def __init__(self, split='train', mode='cbow', C=4):
        super().__init__()
        self.mode = mode
        # 滑动窗口的大小
        self.window_size = 2 * C + 1
        # 加载词库
        self.voc = torch.load('voc/vocab.voc')
        file_paths = [
            'data/wikitext2/train-00000-of-00001.parquet',
            # 'data/wikitext103/train-00000-of-00002.parquet',
            # 'data/wikitext103/train-00001-of-00002.parquet',
        ] if split == 'train' else \
            [
                'data/wikitext2/test-00000-of-00001.parquet',
                # 'data/wikitext103/test-00000-of-00001.parquet',
            ] if split == 'test' else [
                'data/wikitext2/validation-00000-of-00001.parquet',
                # 'data/wikitext103/validation-00000-of-00001.parquet',
            ]

        file_data_list = []
        # 加载数据集
        for file_path in file_paths:
            file_data_list.append(pd.read_parquet(file_path))

        df = pd.concat(file_data_list)

        # 上下文列表
        self.context_list = []
        # 中间词列表
        self.mid_list = []

        for i, row in df.iterrows():
            text = row['text']
            # 预处理
            text = text.lower().strip()
            # 跳过空行或标题
            if text == '' or (text.startswith('= =') and text.endswith('= =')):
                continue

            words = text.split(' ')

            # 跳过文本长度不足，不包含下文和中间词的文本
            if len(words) < C + 2:
                continue
            # 若句子长度大于等于 C + 2 且 小于窗口大小，则填充到窗口大小
            elif len(words) < self.window_size:
                words = words + ['<pad>' for i in range(self.window_size - len(words))]

            # 窗口滑动的次数
            iter_count = len(words) - self.window_size + 1
            for i in range(iter_count):
                # 去对应窗口的文本
                win_words = words[i: i + self.window_size]
                # 获取上下文和中间词
                # 上文
                pre = win_words[:C]
                # 中间词
                mid = win_words[C]
                # 下文
                suf = win_words[C + 1:]
                self.context_list.append(pre + suf)
                self.mid_list.append([mid])

        # print(self.context_list)
        # print(self.mid_list)

    def __len__(self):
        return len(self.mid_list)

    def __getitem__(self, idx):
        inp = self.context_list[idx] if self.mode == 'cbow' else self.mid_list[idx]
        label = self.mid_list[idx] if self.mode == 'cbow' else self.context_list[idx]
        # 文字转索引
        inp = self.voc.lookup_indices(inp)
        label = self.voc.lookup_indices(label)
        return torch.tensor(inp), torch.tensor(label)


if __name__ == '__main__':
    ds = WikiDataset()
    print(len(ds))
    print(ds[0])
