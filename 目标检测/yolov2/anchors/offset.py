import torch


def corners2centers(corners):
    x1 = corners[:, 0::4]
    y1 = corners[:, 1::4]
    x2 = corners[:, 2::4]
    y2 = corners[:, 3::4]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    return torch.concatenate([cx, cy, w, h], dim=-1)


def centers2corners(centers):
    cx = centers[:, 0::4]
    cy = centers[:, 1::4]
    w = centers[:, 2::4]
    h = centers[:, 3::4]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.concatenate([x1, y1, x2, y2], dim=-1)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def offset_boxes(anchors, gts, eps=1e-6):
    c_anchors = corners2centers(anchors)
    c_gts = corners2centers(gts)
    offset_xy = -torch.log(1 / (c_gts[:, :2] - c_anchors[:, :2]) - 1)
    offset_wh = torch.log(c_gts[:, 2:] / (c_anchors[:, 2:] + eps))
    return torch.cat([offset_xy, offset_wh], dim=1)


def offset_inverse(anchors, target_offset):
    c_anchors = corners2centers(anchors)
    c_predict_xy = sigmoid(target_offset[:, :2]) + c_anchors[:, :2]
    c_predict_wh = c_anchors[:, 2:] * torch.exp(target_offset[:, 2:])
    c_predict = torch.cat([c_predict_xy, c_predict_wh], dim=1)
    return centers2corners(c_predict)
