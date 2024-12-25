import torch


def corner2center(box):
    # box 大小 [n,4]
    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = box[:, 2]
    y2 = box[:, 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1).int()


def center2corner(box):
    # box 大小 [n,4]
    cx = box[:, 0]
    cy = box[:, 1]
    w = box[:, 2]
    h = box[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1).int()
