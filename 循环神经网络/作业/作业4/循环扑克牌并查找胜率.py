# 创造一幅扑克牌
# 扑克牌有2个特征: 数字，花色
# 循环数字
# for i in range(2, 15):
#     print(i)
#     # 循环花色
#     for j in range(4):
#         print(j)

cards = [[i, j] for j in range(4) for i in range(2, 15)]
print(cards)

# 数字到扑克牌的映射
card_map = {
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
        print(sorted_cards)
        # 将扑克牌转换成字符串，作为查询的键
        # 获取数字部分的字符串
        # 如果数字在表中存在则返回对应字符，否则返回数字的字符串格式
        s1 = card_map[sorted_cards[0][0]] if sorted_cards[0][0] in card_map else str(sorted_cards[0][0])
        s2 = card_map[sorted_cards[1][0]] if sorted_cards[1][0] in card_map else str(sorted_cards[1][0])
        # 拼接查询用的字符串
        key = f'{s1}{s2}{"" if c1[0] == c2[0] else ("s" if c1[1] == c2[1] else "o")}'
        print(key)
