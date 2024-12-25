import torch

a = torch.tensor([
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1]
])

b = torch.tensor([
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1]
])

TP = (a & b).sum()
F = (a != b).sum()
mIOU = TP / (TP + F)
print(mIOU)

# print((a == b).sum() / ((a == b).sum() + (a != b).sum()))
