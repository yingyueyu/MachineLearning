import torch
from torch import nn

"""
如果外部一直说 device有问题，我们可以就在模型中加入device
"""


class CnnNet(nn.Module):
    def __init__(self, device):
        super(CnnNet, self).__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = device
            raise IOError("设备不可以为空")
        # 3 140 440 作为输入  (140 --> 128 = 12)
        self.conv1 = nn.Conv2d(3, 16, 7, 1, device=device)  # 134 434
        self.conv2 = nn.Conv2d(16, 16, 7, 1, device=device)  # 128 428
        self.conv3 = nn.Conv2d(16, 64, 3, 1, 1, device=device)  # 64 214
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, device=device)  # 32 107
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, device=device)  # 16 54
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, device=device)  # 8 27
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, device=device)  # 4 13

        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4 * 13 * 1024, 1024, device=device)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1024, 65 * 7, device=device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.max_pool(self.relu(self.conv5(x)))
        x = self.max_pool(self.relu(self.conv6(x)))
        x = self.max_pool(self.relu(self.conv7(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 7, 65)
        return self.softmax(x)

# if __name__ == '__main__':
#     model = CNN()
#     images = torch.randn(1, 3, 140, 440)
#     result = model(images)
#     print(result.shape)
