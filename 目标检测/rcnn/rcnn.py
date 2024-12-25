import cv2
import numpy as np
import torch

from backbone.resnet import ResNet
from bbox.selective_search_cv import generate_bbox
from svm.svm import SVMNet
from torchvision.ops import box_iou

if __name__ == '__main__':
    # 此处选择性搜索算法的颜色标准是RGB
    image = cv2.imread("./img/catdog.jpg")
    h, w = image.shape[:2]
    # ss算法生成的anchors
    anchors = generate_bbox(image)
    anchors = np.stack(anchors, axis=0)
    # 将生成的anchors坐标修改为左上角和右下角坐标
    axy, awh = anchors[:, :2], anchors[:, 2:]
    anchors[:, 2:] = axy + awh

    # 实际中标注的框
    ground_truth = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                                 [0.55, 0.2, 0.9, 0.88]])

    features = []
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        feature = image[y1:y2, x1:x2]
        feature = cv2.resize(feature, (224, 224))
        features.append(feature)

    features = np.stack(features, axis=0)
    features = torch.from_numpy(features).permute([0, 3, 1, 2])
    print(features.shape)

    resnet = ResNet()  # resnet如果已经使用预训练参数，则不需要再进行训练
    svm = SVMNet(3)  # 需要单独进行训练
    resnet_result = resnet(features.float())
    svm_result = svm(resnet_result)
    print(svm_result.shape)  # 输入特征

    # 将ground_truth原本 左上角坐标+宽高 --> 左上角点和右下角点
    # x1,y1,w,h => w = x2 - x1  h = y2 - y1
    ground_truth[:, 2:] = ground_truth[:, 2:] + ground_truth[:, :2]
    # 将anchors转化为比例值
    anchors = torch.tensor(anchors)
    anchors = anchors / torch.tensor([w, h, w, h])
    ious = box_iou(anchors, ground_truth)

    print(ious)

    # bg_col = torch.zeros((len(anchors), 1))
    # labels = torch.concatenate([bg_col, ious], dim=-1)
    # # 论文上表述当iou的值大于0.5 即为正样本，当iou的值小于0.3 为负样本（背景、0）
    # labels[labels > 0.3] = 1
    # labels[labels < 0.3] = 0
    # for label in labels:
    #     if label[0] == 0 and label[1] == 0 and label[2] == 0:
    #         label[0] = 1
    # print(labels)  # 输出特征
