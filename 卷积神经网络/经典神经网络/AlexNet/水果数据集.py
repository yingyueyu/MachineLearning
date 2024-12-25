from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import Compose, ToTensor, Resize


class FruitDataset(Dataset):
    # transform: 图片预处理
    # target_transform: 标签预处理
    def __init__(self, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        # 读取元数据
        self.df = pd.read_csv('./data.meta.csv')
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取一行元数据
        row = self.df.iloc[idx]
        # 读取图片
        img_path = row['img_path']
        image = Image.open(img_path)
        # 标签
        label = row['label']
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224), antialias=True)
    ])

    fruit_dict = {
        'Banana': 0,
        'Corn': 1,
        'Watermelon': 2
    }

    ds = FruitDataset(transform, lambda label: fruit_dict[label])
    print(len(ds))
    print(ds[0])
