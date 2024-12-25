import os
import pandas as pd
import re

data = {
    # key: 表头
    # value: 数据
    'img_path': [],
    'label': []
}

# regex_partten = '_(45|90|135|180|225|270|315).jpg$'
#
# # 扫描文件夹
# entries = os.scandir('./data')
# for entry in entries:
#     img_entries = os.scandir(entry.path)
#     for img_entry in img_entries:
#         if re.search(regex_partten, img_entry.name):
#             os.remove(img_entry.path)

entries = os.scandir('./data')
for entry in entries:
    img_entries = os.scandir(entry.path)
    for img_entry in img_entries:
        # 保存图片路径
        data['img_path'].append(img_entry.path)
        # 保存标签
        data['label'].append(entry.name)

df = pd.DataFrame(data)
# 保存 csv 文件
df.to_csv('./data.meta.csv', index=False)
