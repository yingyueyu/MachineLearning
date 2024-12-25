import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


# 扑克牌起手牌胜率数据集
class CardDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 读取元数据
        self.df = pd.read_csv('德州扑克起手牌获胜率.csv')

        self.inputs = []
        self.labels = []

        # 扑克牌
        cards = [[i, j] for j in range(4) for i in range(2, 15)]
        # 数字到扑克牌的映射
        self.card_map = {
            10: 'T',
            11: 'J',
            12: 'Q',
            13: 'K',
            14: 'A'
        }

        # 使用双重for循环遍历所有两张牌的组合
        for i, c1 in enumerate(cards):
            for j, c2 in enumerate(cards):
                # 两张牌相同则跳过
                if i == j:
                    continue
                # 为了查找表格，此处做一个从大到小的排序
                sorted_cards = sorted([c1, c2], key=lambda c: c[0], reverse=True)
                # print(sorted_cards)
                # 将扑克牌转换成字符串，作为查询的键
                # 获取数字部分的字符串
                # 如果数字在表中存在则返回对应字符，否则返回数字的字符串格式
                s1 = self.card_map[sorted_cards[0][0]] if sorted_cards[0][0] in self.card_map else str(
                    sorted_cards[0][0])
                s2 = self.card_map[sorted_cards[1][0]] if sorted_cards[1][0] in self.card_map else str(
                    sorted_cards[1][0])
                # 拼接查询用的字符串
                key = f'{s1}{s2}{"" if c1[0] == c2[0] else ("s" if c1[1] == c2[1] else "o")}'
                # 查找对应 key 的数据行
                row_num = self.df.index[self.df['起手牌'] == key]
                row_data = self.df.iloc[row_num]
                # 查询胜率和平局率
                win = float(row_data['获胜概率'].values[0][:-1]) * 0.01
                draw = float(row_data['平局概率'].values[0][:-1]) * 0.01

                # 保存扑克牌
                self.inputs.append([c1, c2])
                # 保存标签
                self.labels.append([win, draw])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        label = torch.tensor(self.labels[idx])
        # 对扑克牌做 one-hot 编码
        # 对数字部分进行编码
        num = F.one_hot(torch.tensor(inp)[:, 0], 15)
        # 对花色进行编码
        face = F.one_hot(torch.tensor(inp)[:, 1], 4)
        # 连接数字和花色
        inp = torch.cat([num, face], dim=1)
        return inp, label


if __name__ == '__main__':
    ds = CardDataset()
    print(len(ds))
    print(ds[0])
    print(ds[0][0].shape)
