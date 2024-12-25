# 基于GAN网络，可以将原本Linear改造为Conv2d
import torch
from torch import nn
from tqdm import tqdm
import torchvision
from torchvision import transforms


def cbl(in_channels, out_channels, device):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, device=device),
        nn.BatchNorm2d(out_channels, device=device),
        nn.LeakyReLU()
    )


# 生成网络
class GNet(nn.Module):
    # 假设输入 (batch_size 1 7 7) -> (batch_size 1 28 28)
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.b1 = cbl(1, 32, device)
        self.b2 = cbl(32, 64, device)
        self.b3 = cbl(64, 128, device)

        # nearest 近邻插值法  bilinear 双线性插值法
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.fc = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1, device=device),
            nn.Tanh()  # 选择 -1 ~ 1 因为在Dataset的transform.Normalise的函数（图像正则化结果-1~1）
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.up_sample(self.b1(x))
        x = self.up_sample(self.b2(x))
        x = self.b3(x)
        x = self.fc(x)
        return x


# 判别网络
class DNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, device=device),
            nn.BatchNorm2d(32, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, device=device),
            nn.BatchNorm2d(128, device=device),
            nn.LeakyReLU(),
            # 自适应平均池化，最后一层有多大就设置多大的池化卷积核
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1, device=device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.layers(x)
        return x


if __name__ == '__main__':

    device = torch.device("cuda")
    # 生成网络
    g_model = GNet(device)
    # 判别网络
    d_model = DNet(device)

    # 损失函数 因为两者的损失都可以使用
    criterion = nn.MSELoss()

    # 生成网络与判别网络优化不同，因此优化器有两个
    g_optimizer = torch.optim.Adam(g_model.parameters(), 1e-2)
    d_optimizer = torch.optim.Adam(d_model.parameters(), 1e-2)

    # 数字的MNIST数据集
    batch_size = 100
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
    g_input = torch.normal(0, 1, (batch_size, 1, 7, 7))
    # 生图像的标签
    labels_f = torch.zeros((batch_size, 1), device=device)  # 假图标签
    labels_t = torch.ones((batch_size, 1), device=device)  # 真图标签
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
