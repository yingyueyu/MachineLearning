import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.features = data['features']
        self.labels = data['labels']

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


train_dataset = CustomDataset(torch.load("./voc_classes_car_train.pth", weights_only=False))
valid_dataset = CustomDataset(torch.load("./voc_classes_car_val.pth", weights_only=False))
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=20)

features, labels = next(iter(train_loader))
print(features.shape)
