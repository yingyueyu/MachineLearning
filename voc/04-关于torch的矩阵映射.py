import torch

a = torch.tensor([0, 0, 1, 0])

b = torch.tensor([
    [1, 2, 1],
    [1, 2, 2],
    [1, 2, 1]
])

print(a[b])
