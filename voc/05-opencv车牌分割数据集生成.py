import cv2
from random import randint  # 按要求生成随机整数
import numpy as np
import torch
import os

# 取出一部分VOC中图像数据集作为车牌的背景
file_names = os.listdir("../voc/VOCdevkit/VOC2007/JPEGImages")
# 目标数据
labels = []
cnn_images = []
seg_images = []
seg_labels = []
for i in range(5000):
    # 取出背景
    bg = cv2.imread("./panels/bg/blue_140.PNG")
    # 缩放车牌背景的大小到标注值
    bg = cv2.resize(bg, (440, 140))
    # 所有的车牌字符
    panels = []
    # 随机生成城市字符
    city_element = randint(0, 30)
    panels.append(city_element)  # 往列表中存数据
    # 随机生成城市区域字母
    letter_element = randint(41, 64)
    panels.append(letter_element)  # 往列表中存数据
    # 随机生成剩下的5个字符
    for _ in range(5):
        chr_element = randint(31, 64)
        panels.append(chr_element)  # 往列表中存数据
    # 存放车牌字符的数据序列
    labels.append(panels)
    # 设置横坐标放置的初始值
    start = 20
    # 生成车牌 enumerate将遍历的内容增加一个序列号
    for i, panel_chr in enumerate(panels):
        # 加载对应的图片
        city = cv2.imread(f"./panels/detect/{str(panel_chr)}.jpg")
        # 缩放车牌字体大小
        city = cv2.resize(city, (40, 80))
        # 将车牌的字体调整为黑底白字
        city = 255 - city
        # 将字体放置到界面上
        mask = city > bg[30:110, start:40 + start]
        bg[30:110, start:40 + start][mask] = city[mask]
        # start 每次生成图像时，就网友移动一段距离
        if i == 1:
            start += 40 + 40  # 40 是每个符号的宽度  而40是点的间隔
        else:
            start += 40 + 12  # 40 是每个符号的宽度  而10是两两字之间的间隔
    # TODO 保存车牌用于训练CNN  设置一个保存的路径
    # cv2.imwrite(f"./data/cnn/images/{i}.jpg", bg)
    cnn_images.append(torch.tensor(bg).permute([2, 0, 1]))
    # TODO 将车牌用于训练Unet
    # 缩小车牌大小
    scale = 140 / 440
    new_w = randint(100, 200)
    # 将生成的车牌信息
    blank = np.zeros((448, 448, 3), dtype=np.uint8)
    binary = blank.copy()
    # 生成图像在大图中的位置（防止生成图像越界）
    loc_x = int((448 - new_w) / 2)
    loc_y = int((448 - new_w * scale) / 2)
    loc_w = new_w
    loc_h = int(new_w * scale)
    panel_img = cv2.resize(bg, (loc_w, loc_h))
    # 纯白区域
    panel_white = np.ones((loc_h, loc_w, 3), dtype=np.uint8) * 255
    # 融合图像
    blank[loc_y:loc_y + loc_h, loc_x:loc_x + loc_w] = panel_img
    binary[loc_y:loc_y + loc_h, loc_x:loc_x + loc_w] = panel_white
    # 将这副图进行随机仿射、旋转（缩放）变化
    pts1 = np.float32([
        [loc_x, loc_y],
        [loc_x + loc_w, loc_y],
        [loc_x + loc_w, loc_y + loc_h]
    ])

    rand_offset_x = randint(-50, 50)
    rand_offset_y = randint(-50, 50)
    pts2 = np.float32([
        [loc_x + rand_offset_x, loc_y + rand_offset_y],
        [loc_x + loc_w, loc_y],
        [loc_x + loc_w, loc_y + loc_h]
    ])

    M = cv2.getAffineTransform(pts1, pts2)
    blank = cv2.warpAffine(blank, M, blank.shape[:2])
    binary = cv2.warpAffine(binary, M, blank.shape[:2])

    # 找个实际场景作为背景
    n = randint(0, 1000)
    real_back = cv2.imread(f"../voc/VOCdevkit/VOC2007/JPEGImages/{file_names[n]}")
    real_back = cv2.resize(real_back, (448, 448))
    panel_area = 255 - binary
    panel_area = cv2.bitwise_and(real_back, panel_area)
    blank = cv2.bitwise_or(blank, panel_area)

    # 保存图片 (一定要做延时处理)
    # cv2.imwrite(f"./data/seg/images/{i}.jpg", blank)
    # cv2.imwrite(f"./data/seg/labels/{i}.jpg", binary)
    # binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    binary[binary == 255] = 1
    binary = binary[:, :, 0]
    seg_images.append(torch.tensor(blank).permute([2, 0, 1]))
    seg_labels.append(torch.tensor(binary))
torch.save(labels, "../语义分割/car_panel_detect/data/cnn/labels.pth")
torch.save(cnn_images, "../语义分割/car_panel_detect/data/cnn/cnn_images.pth")
# torch.save(seg_images, "../语义分割/car_panel_detect/data/seg/seg_images.pth")
# torch.save(seg_labels, "../语义分割/car_panel_detect/data/seg/seg_labels.pth")
