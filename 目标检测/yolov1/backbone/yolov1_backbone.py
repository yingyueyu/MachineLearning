import torch
from torch import nn


class YOLOv1Net(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv1Net, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1)
        )
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1)
        )
        self.bn4 = nn.BatchNorm2d(1024)

        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 2, 1)
        )
        self.bn5 = nn.BatchNorm2d(1024)

        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, 1024, 3, 1, 1)
        )
        self.bn6 = nn.BatchNorm2d(1024)

        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (10 + num_classes))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x.view(-1, 7, 7, (10 + self.num_classes))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLOv1Net(2)
    model = model.to(device)
    image = torch.randn(1, 3, 448, 448)
    result = model(image.to(device))

    print(result.shape)
