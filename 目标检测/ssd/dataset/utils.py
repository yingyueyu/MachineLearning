import torch
import numpy as np
from torchvision.ops import box_iou
from dataset.offset import offset_boxes

image_sizes = torch.tensor([[21, 45], [45, 99], [99, 153], [153, 207], [207, 261], [261, 315]])
image_ratios = torch.tensor([1., 0.5, 2, 1 / 3, 3.])


def generated_offsets(image_sizes, image_ratios, num_conv):
    image_size = image_sizes[num_conv - 1, :]
    image_ratio = image_ratios
    if num_conv in [1, 5, 6]:
        image_ratio = image_ratios[:3]
    anchors = []
    for ratio in image_ratio:
        h = (image_size[0] ** 2 / ratio) ** 0.5
        w = h * ratio
        anchors.append(torch.stack([-w, -h, w, h]) / 2)
    default_size = (image_size[0] * image_size[1]) ** 0.5
    anchors.append(torch.stack([-default_size, -default_size, default_size, default_size]) / 2)
    return anchors


def generated_anchors_per_conv(stride, conv_size, offsets):
    x = torch.from_numpy(np.arange(0, conv_size))
    y = torch.from_numpy(np.arange(0, conv_size))
    x, y = torch.meshgrid([x, y], indexing='ij')
    x = x.reshape(-1, 1) * stride
    y = y.reshape(-1, 1) * stride
    points = torch.cat([x, y, x, y], dim=-1)
    point_list = []
    for offset in offsets:
        point_list.append(points + offset)
    points = torch.concatenate(point_list, dim=0)
    return points


def generated_anchors():
    offsets1 = generated_offsets(image_sizes, image_ratios, 1)
    offsets2 = generated_offsets(image_sizes, image_ratios, 2)
    offsets3 = generated_offsets(image_sizes, image_ratios, 3)
    offsets4 = generated_offsets(image_sizes, image_ratios, 4)
    offsets5 = generated_offsets(image_sizes, image_ratios, 5)
    offsets6 = generated_offsets(image_sizes, image_ratios, 6)
    conv_1_anchors = generated_anchors_per_conv(8, 38, offsets1)
    conv_2_anchors = generated_anchors_per_conv(16, 19, offsets2)
    conv_3_anchors = generated_anchors_per_conv(30, 10, offsets3)
    conv_4_anchors = generated_anchors_per_conv(60, 5, offsets4)
    conv_5_anchors = generated_anchors_per_conv(100, 3, offsets5)
    conv_6_anchors = generated_anchors_per_conv(300, 1, offsets6)
    anchors = torch.concatenate([conv_1_anchors,
                                 conv_2_anchors,
                                 conv_3_anchors,
                                 conv_4_anchors,
                                 conv_5_anchors,
                                 conv_6_anchors], dim=0)
    anchors = torch.clamp(anchors, min=0, max=300)
    return anchors


if __name__ == '__main__':
    anchors = generated_anchors()
    ground_truth = torch.tensor([
        [0, 23, 76, 88, 89]])

    ious = box_iou(anchors.int(), ground_truth[:, 1:])
    classes_confidence = torch.max(ious, dim=-1).values
    bg_confidence = torch.zeros(classes_confidence.shape)
    confidence = torch.stack([bg_confidence, classes_confidence], dim=0).reshape(-1, 2)

    reg = offset_boxes(anchors, ground_truth)
    labels = torch.cat([confidence, reg], dim=-1)
    print(labels.shape)
