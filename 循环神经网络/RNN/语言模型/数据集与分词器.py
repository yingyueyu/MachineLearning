import torch
import torchtext
from torch import nn
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from collections import OrderedDict, Counter
import re


# 安装 torchtext
# conda install -c pytorch torchtext=0.18.0

# print(torchtext.__version__)


# 分词器
# 功能:
# 1. split 将字符串分词
# 2. encode 将分词后的结果进行索引编码
# 3. decode 将索引重新解码成词汇

# split
# 'hello, how do you do?'
# ['hello', ',', 'how', 'do', 'you', 'do', '?'] token
#
# encode
# 'hello, how do you do?'
# [0, 1, 2, 3, 4, 3, 6]
#
# decode
# [0, 1, 2, 3, 4, 3, 6]
# ['hello', ',', 'how', 'do', 'you', 'do', '?']

class Tokenizer:
    def __init__(self):
        # 创建一个词汇表，用于存储词汇对应的索引关系
        tokens = self._split('how are you? i am fine, thank you.')
        # 统计单词出现次数
        counter = Counter(tokens)
        # 按照单词出现次数排序
        sorted_dict = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        # 有序字典
        od = OrderedDict(sorted_dict)
        # 使用 vocab 建立词库
        # ordered_dict: 有序字典，通常我们把高频词汇排前面
        # min_freq: 最低词频，当词语出现的次数小于 min_freq，则舍弃掉
        # specials: 特殊token，通常是一些有含义的符号
        # special_first: 特殊token是否排列到最前面
        # 特殊符号:
        # <unk>: 未知字符，当编码时出现了当前词库中不存在的字时，则返回 <unk>
        # <pad>: 填充占位
        # <sos> 或 <bos>: 句子的开头
        # <eos>: 句子的结尾
        # <mask>: 掩码
        # <cls>: class 开始分类
        self.voc = vocab(od, min_freq=0, specials=['<unk>', '<pad>', '<sos>', '<eos>', '<mask>', '<cls>'],
                         special_first=True)
        # 设置默认索引，当查找词库中找不到的词时，就返回默认索引
        self.voc.set_default_index(self.voc['<unk>'])

    def __len__(self):
        return len(self.voc)

    # 分割字符串
    # txt: 文本参数
    def _split(self, txt):
        regex = r'<unk>|<pad>|<sos>|<eos>|<mask>|<cls>|\w+|[^\w\s]'
        return re.findall(regex, txt)

    # 编码
    # txt: 文本参数
    def encode(self, txt):
        # 切分
        split = self._split(txt)
        # 查找对应索引
        return self.voc.lookup_indices(split)

    # 解码
    # indices: 索引
    def decode(self, indices):
        return self.voc.lookup_tokens(indices)


class WordDataset(Dataset):
    def __init__(self, C=5):
        super().__init__()
        # 分词器
        self.tokenizer = Tokenizer()
        # 所有文本组成一句用于训练的话
        all_text = '<sos> how are you? i am fine, thank you. <eos>'
        self.words = self.tokenizer.encode(all_text)

        self.inputs = []
        self.labels = []

        # 计算迭代次数
        iter_count = len(self.words) - C
        for i in range(iter_count):
            # 获取输入
            inp = self.words[i:i + C]
            # 标签
            label = self.words[i + 1: i + C + 1]
            self.inputs.append(inp)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        # one_hot 编码
        inp = nn.functional.one_hot(torch.tensor(inp), len(self.tokenizer)).to(torch.float)
        label = torch.tensor(self.labels[idx])
        return inp, label


if __name__ == '__main__':
    # tokenizer = Tokenizer()
    # print(len(tokenizer))

    # 词库 voc 的 api
    # voc 的类型: torchtext.vocab.Vocab
    # api 文档: https://pytorch.org/text/stable/vocab.html?highlight=torchtext+vocab#torchtext.vocab.Vocab

    # # 查看词库的长度
    # print(len(tokenizer.voc))
    #
    # # 查找某个token的索引
    # print(tokenizer.voc['you'])
    # print(tokenizer.voc['both'])
    #
    # # 查找多个单词对应的索引
    # indices = tokenizer.voc.lookup_indices(['i', 'am', 'fine'])
    # print(indices)
    #
    # # 通过索引查询 token
    # tokens = tokenizer.voc.lookup_tokens(indices)
    # print(tokens)

    # indices = tokenizer.encode('how are you?')
    # print(indices)
    # tokens = tokenizer.decode(indices)
    # print(tokens)

    ds = WordDataset()
    print(len(ds))
    print(ds[0])
