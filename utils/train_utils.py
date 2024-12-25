import json
import os

import torch
from torch.utils.data import random_split, DataLoader


# 数据分割器
# 用于自动分割数据集获取训练集和验证集
# 控制一定周期内使用相同的数据集，超过一定周期后，我们重新随机数据集
class DatasetSpliter:
    def __init__(self, ds, batch_size, threshold=10, train_ds_rate=0.8):
        # train_ds_rate: 训练集的比例
        self.ds = ds
        self.batch_size = batch_size
        self.ds_len = len(self.ds)
        self.train_len = int(self.ds_len * train_ds_rate)
        self.val_len = self.ds_len - self.train_len
        # 阈值
        self.threshold = threshold
        # 当前已经调用 get_ds 的次数
        self.count = 0
        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None
        # 初始化数据集
        self._random_split_ds()

    # 获取数据集
    def get_ds(self):
        # 判断当前获取数据集是第几次获取
        if self.count >= self.threshold:
            # 若大于了获取阈值，则重新随机分割数据集
            self.count -= self.threshold
            self._random_split_ds()
        # 统计调用 get_ds 的次数
        self.count += 1
        # 返回分割后的数据集
        return self.train_ds, self.val_ds, self.train_dl, self.val_dl

    # 随机数据集
    def _random_split_ds(self):
        self.train_ds, self.val_ds = random_split(self.ds, [self.train_len, self.val_len])
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)


def save(_save_dir, _model, _loss, _epoch, _lr):
    state_dict = _model.state_dict()
    model_name = _model.__class__.__name__
    file_name = f'{model_name}_{_epoch}.pt'
    # os.path.join 拼接路径
    save_path = os.path.join(_save_dir, file_name)
    torch.save(state_dict, save_path)
    meta_path = os.path.join(_save_dir, 'meta.json')
    # 先读取元数据，若存在数据就追加数据，不存在数据就初始化数据
    try:
        file = open(meta_path, 'r')
        meta = json.load(file)
        file.close()
    except:
        # 初始化meta信息
        meta = {
            'loss': [],
            'epoch': [],
            'lr': []
        }
    # 追加元数据
    meta['loss'].append(_loss)
    meta['epoch'].append(_epoch)
    meta['lr'].append(_lr)

    with open(meta_path, 'w') as file:
        json.dump(meta, file)


# model_class: 模型的类型
# loaded_epoch: 想要加载那个周期的模型
# params: 模型参数
def load(_save_dir, model_class, params=None, loaded_epoch=None):
    if params is None:
        params = {}
    _model = model_class(**params)
    _loss = float('inf')
    _epoch = 0
    _lr = 1e-2

    meta_path = os.path.join(_save_dir, 'meta.json')

    try:
        with open(meta_path, 'r') as file:
            meta = json.load(file)

        _loss = meta['loss'][-1]
        _epoch = meta['epoch'][-1]
        _lr = meta['lr'][-1]

        if loaded_epoch is not None:
            try:
                # 若读取对应 epoch 数据成功，则赋值对应索引的其他参数
                idx = meta['epoch'].index(loaded_epoch)
                _loss = meta['loss'][idx]
                _epoch = meta['epoch'][idx]
                _lr = meta['lr'][idx]
            except:
                pass

        model_name = model_class.__name__
        # 加载最新的模型
        file_name = f'{model_name}_{_epoch}.pt'
        save_path = os.path.join(_save_dir, file_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # map_location: 将训练好的模型参数映射到对应设备中
        state_dict = torch.load(save_path, weights_only=True, map_location=device)
        _model.load_state_dict(state_dict)
        print('模型加载成功')
    except:
        meta = {
            'loss': [],
            'epoch': [],
            'lr': []
        }
        print('模型加载失败')

    return _model, _loss, _epoch, _lr, meta
