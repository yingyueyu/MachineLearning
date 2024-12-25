import torch
from torch import nn
import matplotlib.pyplot as plt

x = torch.linspace(0, 4, 50)
y = torch.cos(x) * x

dropout1, dropout2 = 0.5, 0.5


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.fc1(X))
        X = dropout_layer(X, dropout1)
        X = self.relu(self.fc2(X))
        X = dropout_layer(X, dropout2)
        return self.fc3(X)


model = MyModel()
criterion = nn.MSELoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.05)
epochs = 100
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
losses = []
for epoch in range(epochs):
    sgd.zero_grad()
    y_predict = model(x)
    loss = criterion(y_predict, y)
    print(loss.item())
    losses.append(loss.item())
    loss.backward()
    sgd.step()

plt.plot(torch.arange(len(losses)), losses, 'r-')
plt.show()
