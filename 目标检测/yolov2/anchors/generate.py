import numpy as np
import cv2
import torch
from torchvision.ops import box_iou
from anchors.kmeans import k_means
from anchors.offset import offset_boxes, corners2centers, centers2corners


def draw_corners(image, anchors_corners, S, B):
    anchors = anchors_corners.reshape(S, S, B * 4)
    for i in range(S):
        for j in range(S):
            anchor = anchors[i, j, :]
            a1 = anchor[:4]
            a2 = anchor[4:]
            cv2.rectangle(image, a1[:2], a1[2:], color=(0, 0, 255), thickness=1)
            cv2.rectangle(image, a2[:2], a2[2:], color=(0, 0, 255), thickness=1)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def generate_anchor_base(anchors_sizes=(64, 128, 256), anchors_ratios=(0.5, 1., 2.)):
    anchors_offsets = []
    for anchors_size in anchors_sizes:
        for anchors_ratio in anchors_ratios:
            h = (anchors_size ** 2 / anchors_ratio) ** 0.5
            w = h * anchors_ratio
            anchors_offsets.append([-w, -h, w, h])
    return torch.tensor(anchors_offsets) / 2


def _enumerate_shifted_anchor(offsets, stride, width, height):
    """
    生成原图上所有对应特征位置的anchors
    :param offsets: generate_anchor_base 每个中心点坐标生成的9个anchors偏移量
    :param stride: 卷积之后的小图映射到大图的步长（缩放的倍率大小）
    :param width:  卷积后的特征图的宽度
    :param height: 卷积后的特征图的高度
    :return:
    """
    anchors = []
    for i in range(width):
        for j in range(height):
            img_x = i * stride
            img_y = j * stride
            loc_anchors = offsets + torch.tensor([img_x, img_y, img_x, img_y])
            # 限制anchors不可以越界
            loc_anchors[:, :2] = torch.clamp(loc_anchors[:, :2], min=0)
            loc_anchors[:, 2] = torch.clamp(loc_anchors[:, 2], max=width * stride)
            loc_anchors[:, 3] = torch.clamp(loc_anchors[:, 3], max=height * stride)

            loc_anchors = corners2centers(loc_anchors)
            anchors.append(loc_anchors)

    anchors = torch.stack(anchors, dim=0).view(-1, 4)
    return anchors


def _generate_confidence_anchors(anchors, ground_truth):
    box1 = centers2corners(anchors)
    box2 = centers2corners(ground_truth[:, 1:])
    confidence = box_iou(box1, box2)
    anchors = torch.concatenate([confidence, anchors], dim=-1)
    return anchors


def _generate_classes_anchors(anchors, ground_truth, num_classes):
    # 在维度中加入分类的区间
    box1 = centers2corners(anchors[:, 1:])
    box2 = centers2corners(ground_truth[:, 1:])
    ious = box_iou(box1, box2).numpy()
    ious[ious >= 0.3] = 1
    ious[ious < 0.3] = 0

    ious = torch.tensor(ious)
    classes = torch.concatenate([1 - ious, ious], dim=-1)
    return torch.concatenate([anchors, classes], dim=-1)


def _generate_anchors(image, ground_truth, num_classes):
    h, w = image.shape[:2]
    # 采用Faster-RCNN 方式生成anchors
    anchors_offsets = generate_anchor_base()
    bbox_anchors = _enumerate_shifted_anchor(anchors_offsets, 32, 13, 13)
    # TODO 通过IOU方式找到IOU值比较大的anchors IOU > 0.3
    # 利用k-means距离找到符合要求的5个anchors
    bbox_anchors = torch.from_numpy(k_means(bbox_anchors.numpy(), 5))
    bbox_anchors = bbox_anchors / torch.tensor([[w, h, w, h]])
    # 在 anchors 中加入置信度
    confidence_anchors = _generate_confidence_anchors(bbox_anchors, ground_truth)
    anchors = _generate_classes_anchors(confidence_anchors, ground_truth, num_classes)
    # 设置为偏移参数
    anchors[:, 1:5] = offset_boxes(anchors[:, 1:5], corners2centers(ground_truth[:, 1:]))
    return anchors


if __name__ == '__main__':
    image = cv2.imread("../imgs/test.jpg")
    ground_truth = torch.tensor([[0, 0.552455, 0.514509, 0.877232, 0.685268]])
    anchors = _generate_anchors(image, ground_truth, 2)
    print(anchors)
