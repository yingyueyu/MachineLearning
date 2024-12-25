import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
import pandas as pd
from collections import Counter, OrderedDict
import torch


# 创建词库
# file_paths: 文件列表
def make_vocab(file_paths, min_freq=50):
    file_data_list = []

    # 保存词库
    voc_map = {}

    # 加载数据文件
    for file_path in file_paths:
        file_data_list.append(pd.read_parquet(file_path))

    # 合并数据
    df = pd.concat(file_data_list)

    # 读取每一行数据
    for i, row in df.iterrows():
        # 获取一行的文本
        text = row['text']
        # 预处理文本
        text = text.lower().strip()
        # 文本若是空行就跳过
        if text == '':
            continue
        # 分割文本
        words = text.split(' ')
        # 统计单词
        counter = Counter(words)
        # k: 文本
        # v: 出现次数
        for k, v in counter.items():
            if k in voc_map:
                # 累计次数
                voc_map[k] += v
            else:
                voc_map[k] = v

    # 按照出现次数排序
    sorted_items = sorted(voc_map.items(), key=lambda item: item[1], reverse=True)
    od = OrderedDict(sorted_items)

    # 创建词库
    voc = vocab(od, min_freq=min_freq, specials=['<unk>', '<pad>', '<sos>', '<bos>', '<eos>', '<mask>', '<cls>'],
                special_first=True)
    voc.set_default_index(voc['<unk>'])

    # 保存词库
    torch.save(voc, 'voc/vocab.voc')


if __name__ == '__main__':
    # make_vocab([
    #     'data/wikitext2/test-00000-of-00001.parquet',
    #     'data/wikitext2/train-00000-of-00001.parquet',
    #     'data/wikitext2/validation-00000-of-00001.parquet',
    #     'data/wikitext103/test-00000-of-00001.parquet',
    #     'data/wikitext103/train-00000-of-00002.parquet',
    #     'data/wikitext103/train-00001-of-00002.parquet',
    #     'data/wikitext103/validation-00000-of-00001.parquet',
    # ])
    voc = torch.load('voc/vocab.voc')
    words = 'i love you .'.split()
    indices = voc.lookup_indices(words)
    print(indices)
    tokens = voc.lookup_tokens(indices)
    print(tokens)
