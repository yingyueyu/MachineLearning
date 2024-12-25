from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize

# 总结流程:
# 1. 创建自定义类继承 Dataset
# 2. 读取元数据 meta.csv  xml  json
# 3. 重写 __len__ 返回数据集长度
# 4. 重写 __getitem__ 返回索引对应的数据
#       1. 先找到对应索引的元数据
#       2. 加载图片
#       3. 转换图片为张量，重置图片大小
#       4. 返回图片和标签


class CatDogDataset(Dataset):
    # transform: 转换图片为张量的函数
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        # 读取元数据
        # data_frame = pd.read_csv('./meta.csv')
        # 表格对象
        self.df = pd.read_csv('./meta.csv')
        # 逐行读取表格，缓存内容
        # for i, row in df.iterrows():
        #     print(row['img_path'])
        #     print(row['label'])

        # print(self.df.iloc[0])
        # print(self.df.iloc[0]['img_path'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 从 df 中取出对应索引的元数据
        row = self.df.iloc[idx]
        img_path = row['img_path']
        label = row['label']
        # 读取图片
        image = Image.open(img_path)
        # 判断是否存在转换器
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224), antialias=True)
    ])

    ds = CatDogDataset(transform=transform)
    print(len(ds))
    inp, label = ds[0]
    print(inp.shape)
    print(label)
    # 数据加载器
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    for i, (inputs, labels) in enumerate(dl):
        print(inputs.shape)
        print(labels)
