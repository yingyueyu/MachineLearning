import numpy as np
import cv2
import torch
from torchvision.ops import box_iou


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


def corners2centers(corners):
    x1 = corners[:, 0::4]
    y1 = corners[:, 1::4]
    x2 = corners[:, 2::4]
    y2 = corners[:, 3::4]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return np.concatenate([cx, cy, w, h], axis=-1)


def centers2corners(centers):
    cx = centers[:, 0::4]
    cy = centers[:, 1::4]
    w = centers[:, 2::4]
    h = centers[:, 3::4]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.concatenate([x1, y1, x2, y2], axis=-1)


def _generate_bbox_anchors(S=7, B=2):
    # bounding box 边框坐标信息是 cx,cy,w,h
    # confidence 与真实边框的IOU的值 （原文 IOU * 类别概率）
    anchors_x1y1 = np.random.randint(0, 448, (S * S * B, 2))
    anchors_wh = np.random.randint(0, 448, (S * S * B, 2))
    anchors_x2y2 = anchors_x1y1 + anchors_wh
    anchors_x2y2[anchors_x2y2 > 448] = 448
    anchors_corners = np.concatenate([anchors_x1y1, anchors_x2y2], axis=-1)
    anchors_centers = corners2centers(anchors_corners) / 448.
    return anchors_centers


def _generate_confidence_anchors(anchors, ground_truth):
    box1 = torch.tensor(centers2corners(anchors))
    box2 = torch.tensor(centers2corners(ground_truth[:, 1:]))
    confidence = box_iou(box1, box2).numpy()
    anchors = np.concatenate([confidence, anchors], axis=-1)
    return anchors


def _generate_classes_anchors(anchors, ground_truth, S, num_classes):
    stride = 1 / S
    gt_cx = ground_truth[:, 1]
    gt_cy = ground_truth[:, 2]

    obj_label = np.array([0, 1])

    anchors = anchors.reshape(S, S, -1)
    classes_column = np.zeros((S, S, num_classes))
    anchors = np.concatenate([anchors, classes_column], axis=-1)
    for i, j in zip((gt_cx / stride).astype(np.int32), (gt_cy / stride).astype(np.int32)):
        anchors[i, j, 10:] = obj_label

    pr = np.argmax(anchors[:, :, 10:], axis=-1)
    anchors[:, :, 0] = anchors[:, :, 0] * pr
    anchors[:, :, 5] = anchors[:, :, 5] * pr
    return anchors


def _generate_anchors(image, ground_truth, S, num_classes):
    bbox_anchors = _generate_bbox_anchors()
    confidence_anchors = _generate_confidence_anchors(bbox_anchors, ground_truth)
    anchors = _generate_classes_anchors(confidence_anchors, ground_truth, S, num_classes)
    return anchors


if __name__ == '__main__':
    image = cv2.imread("../imgs/test.jpg")
    ground_truth = np.array([[0, 0.552455, 0.514509, 0.877232, 0.685268]])
    anchors = _generate_anchors(image, ground_truth, 7, 2)
    print(anchors.shape)
