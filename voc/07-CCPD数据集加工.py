import cv2
import numpy as np
import os

import torch

"""
使用CCPD数据集
"""

file_names = os.listdir("./VOCdevkit/ccpd/")
features = []
labels = []
for file_name in file_names[:1000]:
    image = cv2.imread(f"./VOCdevkit/ccpd/{file_name}")
    img_h, img_w = image.shape[:2]
    part_list = file_name.split("-")
    points = part_list[3].split("_")
    rb_point = points[0].split("&")
    lb_point = points[1].split("&")
    lt_point = points[2].split("&")
    rt_point = points[3].split("&")
    points = [rb_point, lb_point, lt_point, rt_point]
    points = np.array([[int(a), int(b)] for a, b in points], np.int32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    image = cv2.resize(image, (224, 224))
    mask = cv2.resize(mask, (224, 224))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    features.append(torch.from_numpy(image).permute([2, 0, 1]))
    labels.append(torch.from_numpy(mask) / 255)

    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask[y1:y2, x1:x2, :])
    # cv2.imshow("image1", image[y1:y2, x1:x2, :])
    # cv2.waitKey(0)
    # torch.contiguous()

torch.save({"features": features, "labels": labels}, "VOCdevkit/panels/ccpd_1k_224.pth")
