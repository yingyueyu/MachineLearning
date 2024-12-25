"""
判断是否为汽车
在"/VOCdevkit/VOC2007/ImageSets/Main" 中存在car_train.txt car_valid.txt
其中，1 代表正样本存在汽车  -1 代表负样本不存在汽车
"""
import os
import cv2
import numpy as np
import torch

class_name = "car"
root = "./VOCdevkit/VOC2007"
image_size = 224


def read_voc_class_images(class_name, mode="train"):
    f_path = os.path.join(root, "ImageSets", "Main", f"{class_name}_{mode}.txt")
    with open(f_path) as f:
        lines = f.readlines()

    features = []
    labels = []
    for line in lines:
        name, label = line.split()
        label = int(label) if int(label) > 0 else 0
        image = cv2.imread(os.path.join(root, "JPEGImages", f"{name}.jpg"))  # w h 3
        image = cv2.resize(image, (image_size, image_size))  # 224 224 3
        image = torch.from_numpy(image).permute([2, 0, 1])  # 3 224 224
        features.append(image)
        labels.append(label)

    labels = torch.tensor(labels)
    return {"features": features, "labels": labels}


train_data = read_voc_class_images(class_name, "train")
valid_data = read_voc_class_images(class_name, "val")
torch.save(train_data, "./voc_classes_car_train.pth", )
torch.save(valid_data, "./voc_classes_car_val.pth", )
