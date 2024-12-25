import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import jieba
from collections import Counter, OrderedDict
from torchtext.vocab import vocab


# import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 读取元数据
        self.df = pd.read_csv('meta.csv')
        self.transform = Compose([
            ToTensor(),
            Resize((224, 224), antialias=True)
        ])
        self.label_map = {
            'Airplane': 0,
            'Car': 1,
            'Ship': 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取对应索引的一行元数据
        meta_data = self.df.iloc[idx]
        # 图片路径
        img_path = meta_data['img_path']
        # 标签
        label = meta_data['label']
        # 转换数据类型
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor(self.label_map[label])
        return image, label


# 分词器
class Tokenizer:
    def __init__(self):
        all_text = '这张图片是，汽车，飞机，轮船。'
        tokens = self._split(all_text)
        # 创建词库
        # 统计
        counter = Counter(tokens)
        # 排序
        sorted_result = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        # 有序字典
        od = OrderedDict(sorted_result)
        # 词库
        self.voc = vocab(od, min_freq=0, specials=['<unk>', '<pad>', '<sos>', '<eos>'], special_first=True)
        self.voc.set_default_index(self.voc['<unk>'])
        # print(f'voc len: {len(self.voc)}')

    # 分词
    def _split(self, txt):
        tokens = [token for token in jieba.cut(txt, cut_all=False)]
        return tokens

    def encode(self, txt):
        txt_list = self._split(txt)
        return self.voc.lookup_indices(txt_list)

    def decode(self, idx):
        return self.voc.lookup_tokens(idx)


# 语言模型数据集
class LangDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()

        self.inputs = [0, 1, 2]
        self.labels = [
            '这张图片是飞机。',
            '这张图片是汽车。',
            '这张图片是轮船。',
        ]
        self.labels = [self.tokenizer.encode(label) for label in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        label = self.labels[idx]
        return torch.tensor([inp], dtype=torch.float), torch.tensor(label)


if __name__ == '__main__':
    # ds = ImageDataset()
    # print(len(ds))
    # print(ds[0])
    # t = Tokenizer()
    # idx = t.encode('这张图片是汽车。')
    # print(idx)
    # txt = t.decode(idx)
    # print(txt)
    ds = LangDataset()
    print(len(ds))
    print(ds[0])
    print(ds.tokenizer.voc['。'])
