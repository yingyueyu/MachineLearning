import torch
from torch import nn
import matplotlib.pyplot as plt

# 加载保存的数据
data = torch.load("./data.pth")
# print(data.shape)
# 将数据集分为 train(训练数据集)  valid(验证数据集)  test(测试数据集)
# train 在每一个epoch计算训练数据集的损失以及准确率
#      （一般来说，训练的损失会下降，准确率会提高，关键点：欠拟合程度）
# valid 在每一个epoch计算验证数据集的损失以及准确率
#      （对整个模型的泛化能力评估，关键点：过拟合程度 （train_acc < valid_acc,train_loss > valid_loss））
# test 对整个模型进行评估（比较模型和模型谁更优，谁更好。）

# train：valid：test = 8：3：1
n = data.shape[0] // 12
train_data = data[:8 * n, :]
valid_data = data[8 * n:11 * n, :]
test_data = data[11 * n:, :]


# print(train_data.shape)
# print(valid_data.shape)
# print(test_data.shape)

# -------模型---------------------------------
class CustomNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))

        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


# -------准确率计算---------------------------
def accuracy(predict, labels):
    out = torch.argmax(predict, dim=-1)
    return sum(out == labels) / labels.shape[0]


# -------图线初始化----------------------------
fig, ax = plt.subplots(1, 1)
train_loss_pts = [1.]
valid_loss_pts = [1.]
train_acc_pts = [0.]
valid_acc_pts = [0.]
line_train_loss, = ax.plot(range(len(train_loss_pts)), train_loss_pts, label="train_loss")
line_valid_loss, = ax.plot(range(len(valid_loss_pts)), valid_loss_pts, label="valid_loss")
line_train_acc, = ax.plot(range(len(train_acc_pts)), train_acc_pts, label="train_acc")
line_valid_acc, = ax.plot(range(len(valid_acc_pts)), valid_acc_pts, label="valid_acc")

# -------训练---------------------------------
train_features = train_data[:, :3]  # 80,3
train_labels = train_data[:, 3]  # 80
valid_features = valid_data[:, :3]  # 30,3
valid_labels = valid_data[:, 3]  # 30
test_features = test_data[:, :3]  # 30,3
test_labels = test_data[:, 3]  # 30

# --- 初始化训练的内容，模型、损失、优化器
model = CustomNet()
criterion = nn.CrossEntropyLoss()
adam = torch.optim.SGD(model.parameters(), lr=0.05)

# --- 开始训练
epochs = 10000
for epoch in range(epochs):
    adam.zero_grad()
    # 训练集预测
    predict_labels = model(train_features)
    # 训练集损失
    train_loss = criterion(predict_labels, train_labels.long())
    # 训练集的准确率
    train_acc = accuracy(predict_labels, train_labels)
    # 验证集的损失和准确率
    with torch.no_grad():
        predict_valid_labels = model(valid_features)
        valid_loss = criterion(predict_valid_labels, valid_labels.long())
        valid_acc = accuracy(predict_valid_labels, valid_labels)
    # 训练集损失的反向传播
    train_loss.backward()
    adam.step()

    # 图线更新
    train_loss_pts.append(train_loss.item())
    valid_loss_pts.append(valid_loss.item())
    train_acc_pts.append(train_acc.item())
    valid_acc_pts.append(valid_acc.item())

    if epoch % 100 == 0:
        line_train_loss.set_data(range(len(train_loss_pts)), train_loss_pts)
        line_valid_loss.set_data(range(len(valid_loss_pts)), valid_loss_pts)
        line_train_acc.set_data(range(len(train_acc_pts)), train_acc_pts)
        line_valid_acc.set_data(range(len(valid_acc_pts)), valid_acc_pts)
        ax.relim()
        ax.autoscale_view()
        plt.legend()
        plt.pause(0.2)

        print(f"epoch {epoch + 1} / {epochs} \t"
              f"--train_loss:{train_loss.item():.4f} "
              f"--train_acc:{train_acc.item():.4f} "
              f"--valid_loss:{valid_loss.item():.4f} "
              f"--valid_acc:{valid_acc.item():.4f} ")

# --- 关于test的最终评估（决定启用哪一个模型） ----
model.eval()  # 启动测试模型（不训练）
predict_test_labels = model(test_features)
test_loss = criterion(predict_test_labels, test_labels.long())
test_acc = accuracy(predict_test_labels, test_labels)
print(f"final --test_loss:{test_loss.item():.4f} --test_acc:{test_acc.item():.4f}")
