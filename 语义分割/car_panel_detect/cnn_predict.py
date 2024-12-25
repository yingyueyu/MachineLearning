import torch
import cv2
import numpy as np
from backbone.cnnnet import CnnNet

idx2chr = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
           "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
           "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
           "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
           "W", "X", "Y", "Z"]

model = CnnNet(torch.device("cpu"))
state = torch.load("./save/best.pth", weights_only=False)
model.load_state_dict(state_dict=state)

model.eval() # 注意卷积模型中的dropout需要使用eval进行屏蔽
# 140x440x3
img = cv2.imread("./test/example.jpg")
img = np.expand_dims(img, 0)
img = torch.from_numpy(img)
img = torch.permute(img, [0, 3, 1, 2])
predict = model(img.float())
result = torch.argmax(predict, dim=-1)

print([idx2chr[index.item()] for index in result[0]])
