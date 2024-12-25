import torch
from torch import nn
from CustomNet import CustomNet

model = CustomNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)

for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False

for epoch in range(10):
    x = torch.randn((3, 8))
    label = torch.randint(0, 10, [3]).long()
    output = model(x)

    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("-" * 10, "trained", "-" * 10)
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)
