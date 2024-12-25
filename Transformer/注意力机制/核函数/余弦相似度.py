import numpy as np
import torch
import torch.nn as nn

A = np.random.randn(10)
B = np.random.randn(10)

# 分母
fm = np.sqrt(np.dot(A, A)) * np.sqrt(np.dot(B, B))

# 分子
fz = np.dot(A, B)

# 余弦相似度
cos = fz / fm

print(cos)

A = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
B = torch.tensor([1, 2, 3, 4, 3, 6], dtype=torch.float)

# pytorch 的余弦相似度模块
cs = nn.CosineSimilarity(dim=0)
cos = cs(A, B)
print(cos)
