import torch
import torch.nn as nn

x = torch.randint(1, 11, (3, 10, 10)).to(torch.float)
dp = nn.Dropout(p=0.5)
y = dp(x)
print(x.numel())
print((y == 0).sum())
