from torch import nn


class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x))