import torch
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm


def dbl(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_features)
    )


class GNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = dbl(100, 256)
        self.b2 = dbl(256, 512)
        self.b3 = dbl(512, 1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.fc(x)
        return x.reshape(-1, 1, 28, 28)


class DNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.reshape(-1)


batch_size = 1000
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 均值和标准差是ImageNet数据集的
])

# 加载MNIST数据集
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

g_model = GNet()
d_model = DNet()

g_optimizer = torch.optim.Adam(g_model.parameters(), 1e-3)
d_optimizer = torch.optim.Adam(d_model.parameters(), 1e-3)
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

is_d_train = True
epochs = 60000
g_input = torch.normal(0, 1, (batch_size, 100))
loss_d, loss_g, loss_d_f, loss_d_t = 0, 0, 0, 0
for epoch in range(epochs):
    print(f"epoch {epoch + 1} / {epochs}")
    loop = tqdm(train_loader)
    g_labels = torch.zeros((batch_size, ))
    labels = torch.ones((batch_size, ))
    for images, _ in loop:
        if is_d_train:
            # 鉴别网络训练开启，生成网络不训练
            d_model.train()
            g_model.eval()

            d_optimizer.zero_grad()
            g_images = g_model(g_input)

            g_labels_pre = d_model(g_images)
            loss_d_f = criterion(g_labels_pre, g_labels)

            labels_pre = d_model(images)
            loss_d_t = criterion(labels_pre, labels)

            loss_d = (loss_d_t + loss_d_f) / 2
            loss_d.backward()
            d_optimizer.step()
            if loss_d <= 0.1:
                is_d_train = False
        else:
            # 鉴别网络训练关闭，生成网络训练
            d_model.eval()
            g_model.train()
            g_optimizer.zero_grad()

            g_images = g_model(g_input)
            g_labels_pre = d_model(g_images)
            loss_g = criterion(g_labels_pre, labels)
            loss_g.backward()
            g_optimizer.step()
            if loss_g <= 0.1:
                is_d_train = True
                torch.save(g_images, f"save/epoch_{epoch + 1}_g_images.pt")
                torch.save(images, f"save/epoch_{epoch + 1}_images.pt")

        loop.set_description(
            f"loss_d:{loss_d:.4f}, loss_g:{loss_g:.4f}, loss_d_f:{loss_d_f:.4f}, loss_d_t:{loss_d_t:.4f}")
