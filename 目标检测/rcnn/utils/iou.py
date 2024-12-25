import torch
from torchvision.ops import box_iou

# def iou_boxes(box1, box2):
#     """
#     计算两个框的IOU
#     :param box1:
#     :param box2:
#     :return:
#     """
#     box_area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
#     # 计算两个区域的面积
#     box1_area = box_area(box1)
#     box2_area = box_area(box2)
#     # 计算IoU
#     points_lt = torch.max(box1[:, None, :2], box2[:, :2])
#     points_rb = torch.min(box1[:, None, 2:], box2[:, 2:])
#     inner_wh = torch.clamp(points_rb - points_lt, min=0)
#     # 如果两个不相交，则这里的面积将会是一个负数的值，因此我们认为交集为0
#     inner_area = inner_wh[:, :, 0] * inner_wh[:, :, 1]
#     # (4,1) + (2,1) - (4,2)
#     # box1_area[:,None] ---> 将此时的box1_area 升维到 4,1,1
#     union = box1_area[:, None] + box2_area - inner_area
#     return inner_area / union


if __name__ == '__main__':
    bbox = torch.tensor([
        [100, 100, 200, 200],
        [150, 100, 200, 150],
        [50, 50, 150, 150],
        [130, 100, 200, 150]
    ])
    gts = torch.tensor([
        [120, 120, 200, 200],
        [120, 100, 180, 150],
    ])
    result = box_iou(bbox, gts)
    print(result)
