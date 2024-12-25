import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding='same')
        self.conv3 = nn.Conv2d(256, 384, 3, padding='same')
        self.conv4 = nn.Conv2d(384, 384, 3, padding='same')
        self.conv5 = nn.Conv2d(384, 256, 3, padding='same')
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.pool = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        # Dropout 是正则化技术值一，用于随机弃用部分数据，抑制部分神经元的表达
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    # x: (N, 3, 224, 224)
    def forward(self, x):
        x = self.conv1(x)
        # N x 96 x 55 x 55
        x = self.relu(x)
        x = self.pool(x)
        # N x 96 x 27 x 27
        x = self.conv2(x)
        # N x 256 x 27 x 27
        x = self.relu(x)
        x = self.pool(x)
        # N x 256 x 13 x 13
        x = self.conv3(x)
        # N x 384 x 13 x 13
        x = self.relu(x)
        x = self.conv4(x)
        # N x 384 x 13 x 13
        x = self.relu(x)
        x = self.conv5(x)
        # N x 256 x 13 x 13
        x = self.relu(x)
        x = self.pool(x)
        # N x 256 x 6 x 6
        x = self.dropout(x)

        x = self.flatten(x)
        # N x 9216

        x = self.fc1(x)
        # N x 4096
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # N x 4096
        x = self.relu(x)
        x = self.fc3(x)
        # N x 1000
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    model = AlexNet()
    x = torch.randn(5, 3, 224, 224)
    y = model(x)
    print(y)
    print(y.shape)
