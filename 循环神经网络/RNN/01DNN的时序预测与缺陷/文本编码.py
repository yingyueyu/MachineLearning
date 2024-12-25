import torch
import torch.nn.functional as F

text = 'hey how are you'
# 字符去重
text = list(set(text))
# print(text)
# 按 ASCii 码排序
text = sorted(text)
print(text)

inp = 'hey h'
label = 'o'

# text.index(c): 查询字符 c 在列表 text 中的索引
idx = [text.index(c) for c in inp]
print(idx)

# 二元词袋编码: 在独热编码基础上，出现那些字符就设置为 1，没出现的字符就是 0

# 独热编码
# 参数一: 索引
# 参数二: 独热编码的长度（总共有多少字）
t = F.one_hot(torch.tensor(idx), len(text))
print(t)
print(t.shape)
t = t.sum(dim=0)
print(t)
# 限制编码大小
t[t > 1] = 1
print(t)

# 编码标签
print(text.index(label))
