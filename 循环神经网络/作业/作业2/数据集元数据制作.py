import os

import pandas as pd

data = {
    # 此处数据描述的是表格中的字段
    "id": [],
    "name": [],
    "label": [],
    'img_path': []
}

id = 0

# 扫描文件夹
entries = os.scandir('data')
for entry in entries:
    img_entries = os.scandir(entry.path)
    for img_entry in img_entries:
        # 为每张图片构造元数据
        data['id'].append(id)
        id += 1
        data['name'].append(img_entry.name)
        data['label'].append(entry.name)
        data['img_path'].append(img_entry.path)

# 创建表格
df = pd.DataFrame(data)

# 保存
df.to_csv('./meta.csv', index=False)
