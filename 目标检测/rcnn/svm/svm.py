import torch
from torch import nn


class SVMNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 如果按照RCNN论文中的描述，则需要fc1，fc2,... fn
        # 此处我们只设计一个fc

        self.fc = nn.Linear(4096, 1024)
        self.fc_f = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.fc_f(x)
        return self.softmax(x)


if __name__ == '__main__':
    inp = torch.randn(13, 4096)
    model = SVMNet(3)
    result = model(inp)
    print(result.shape)

