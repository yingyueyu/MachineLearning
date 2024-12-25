import torch
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm


def cbl(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(out_features)
    )


# 生成网络
class GNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = cbl(100, 256)
        self.b2 = cbl(256, 512)
        self.b3 = cbl(512, 1024)

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


# 判别网络
class DNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    # 生成网络
    g_model = GNet()
    # 判别网络
    d_model = DNet()

    # 损失函数 因为两者的损失都可以使用
    criterion = nn.MSELoss()

    # 生成网络与判别网络优化不同，因此优化器有两个
    g_optimizer = torch.optim.Adam(g_model.parameters(), 1e-3)
    d_optimizer = torch.optim.Adam(d_model.parameters(), 1e-3)

    # 数字的MNIST数据集
    batch_size = 1000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 均值和标准差是ImageNet数据集的
    ])

    # 加载MNIST数据集
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 设置一个两个网络训练状态的开关
    is_generator_train = True
    # 为生成网络生成一则虚拟数据 此处100对应生成网络中的输入100维
    g_input = torch.normal(0, 1, (batch_size, 100))
    # 生图像的标签
    labels_f = torch.zeros((batch_size, 1))  # 假图标签
    labels_t = torch.ones((batch_size, 1))  # 真图标签
    # 生成所有的损失的缓存
    loss_g, loss_d, loss_d_f, loss_d_t = 0, 0, 0, 0
    epochs = 60000
    for epoch in range(epochs):
        print(f"\nepoch: {epoch + 1} / {epochs}\n")
        loop = tqdm(train_loader)
        batch = 0
        for images, _ in loop:
            if is_generator_train:
                # 生成网络的训练
                d_model.eval()  # 冻结判别网络
                g_model.train()  # 开启生成网络
                g_optimizer.zero_grad()
                # 将随机的虚拟数据组成一张图像
                g_images = g_model(g_input)
                # 将生成的图像放入判别网络中进行判别
                g_labels_f = d_model(g_images)
                # 对应真假图进行labels损失计算(让假图越来越像真图)
                loss_g = criterion(g_labels_f, labels_t)
                loss_g.backward()
                g_optimizer.step()
                if loss_g < 0.1:
                    is_generator_train = False
                    torch.save(g_images, f"save/g_images_epoch{epoch + 1}_batch{batch}.pt")
                batch += 1
            else:
                # 判别网络训练
                d_model.train()
                g_model.eval()
                d_optimizer.zero_grad()

                # 先由生成网络生成假图
                g_images = g_model(g_input)
                # 判别网络判断该图像的真伪
                g_labels_f = d_model(g_images)
                # 让判别网络知道生成网络的图是假图
                loss_d_f = criterion(g_labels_f, labels_f)

                # 也需要真图放入判别网络中进行判别
                g_labels_t = d_model(images)
                # 让判别网络知道真实的图，是什么样的
                loss_d_t = criterion(g_labels_t, labels_t)

                # 将上述的损失取平均值即可
                loss_d = (loss_d_t + loss_d_f) / 2
                loss_d.backward()
                d_optimizer.step()
                if loss_d < 0.1:
                    is_generator_train = True

            loop.set_description(
                f"g_loss:{loss_g:.4f}  d_loss:{loss_d:.4f}  loss_d_f:{loss_d_f:.4f}  loss_d_t:{loss_d_t:.4f}")
