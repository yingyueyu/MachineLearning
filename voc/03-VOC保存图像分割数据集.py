"""
判断是否为汽车
在"/VOCdevkit/VOC2007/ImageSets/Main"
其中，1 代表正样本存在汽车  0 代表负样本不存在汽车
"""
import os
import cv2
import numpy as np
import torch

class_names = ["cat", "dog"]
root = "./VOCdevkit/VOC2012"
image_size = 512
# [0, 0, 128] 标记汽车分类的颜色 序号1
# [0, 0, 0] 背景的颜色  序号0
colors = [[0, 0, 0], [0, 0, 128]]


def init_voc_segmentations():
    """
    颜色对照表
    :return:
    """
    map = torch.zeros((256 * 256 * 256,), dtype=torch.int32)
    for i, color in enumerate(colors):
        map[(color[2] * 256 + color[1]) * 256 + color[0]] = i
    return map


def get_voc_labels(image, map):
    """
    分割图与坐标的映射
    :param image: 读取numpy的图像，需要转化为totch.tensor
    :param map: 映射表
    :return: 每一个像素对应的color的类别
    """
    image = torch.from_numpy(image).int()
    colormap = (image[:, :, 2] * 256 + image[:, :, 1]) * 256 + image[:, :, 0]
    return map[colormap]


def read_voc_class_images(class_names, mode="train"):
    features = []
    labels = []
    for i, class_name in enumerate(class_names):
        f_path = os.path.join(root, "ImageSets", "Main", f"{class_name}_{mode}.txt")
        with open(f_path) as f:
            lines = f.readlines()
        for line in lines:
            name, label = line.split()
            label = int(label)
            # 排除所有的负样本的情况,整体数据集会很少
            image_path = os.path.join(root, "JPEGImages", f"{name}.jpg")
            image_seg_path = os.path.join(root, "SegmentationObject", f"{name}.png")
            if label > 0 and os.path.exists(image_path) and os.path.exists(image_seg_path):
                image = cv2.imread(image_path)
                image_seg = cv2.imread(image_seg_path)
                image = cv2.resize(image, (image_size, image_size))
                image = torch.from_numpy(image).permute([2, 0, 1])

                image_seg = cv2.resize(image_seg, (image_size, image_size))
                mask = get_voc_labels(image_seg, init_voc_segmentations()) * (i + 1)

                features.append(image)
                labels.append(mask)

    return {"features": features, "labels": labels}


train_data = read_voc_class_images(class_names, "train")
valid_data = read_voc_class_images(class_names, "val")
torch.save(train_data, "../语义分割/save/voc_object_cat_dog_train.pth", )
torch.save(valid_data, "../语义分割/save/voc_object_cat_log_val.pth", )
