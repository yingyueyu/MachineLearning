from torch import nn


class ChatBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(9, 9)

    def forward(self, x):
        return self.fc(x)
