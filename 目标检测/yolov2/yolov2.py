import cv2
import numpy as np
import torch

from anchors.generate import _generate_anchors
from tqdm import tqdm


def loss_bbox(box1, box2):
    # 中心点位置损失
    box1_xy = box1[:, :, :, [1, 2, 6, 7]]
    box2_xy = box2[:, :, :, [1, 2, 6, 7]]
    loss_xy = torch.mean((box1_xy - box2_xy) ** 2)
    # 尺度损失
    box1_wh = box1[:, :, :, [3, 4, 8, 9]]
    box2_wh = box2[:, :, :, [3, 4, 8, 9]]
    loss_wh = torch.mean((box1_wh ** 0.5 - box2_wh ** 0.5) ** 2)
    loss_loc = loss_xy + loss_wh
    # 置信度损失
    loss_conf1 = torch.mean((box1[:, :, :, [0, 5]] - box2[:, :, :, [0, 5]]) ** 2)
    loss_conf2 = torch.mean((1 - (box1[:, :, :, [0, 5]]) - (1 - box2[:, :, :, [0, 5]])) ** 2)
    # 类别损失
    loss_cls = torch.mean((box1[:, :, :, -2:] - box2[:, :, :, -2:]) ** 2)
    return 5 * loss_loc + loss_conf1 + 0.5 * loss_conf2 + loss_cls


"""
一张图像模拟训练
"""
if __name__ == '__main__':
    image = cv2.imread("./imgs/test.jpg")
    image_tr = torch.from_numpy(image).permute([2, 0, 1])
    ground_truth = np.array([[0, 0.552455, 0.514509, 0.877232, 0.685268]])

    features = image_tr.unsqueeze(0)  # 为输入图像加入一个batch_size=1,输入到CNN中
    labels = _generate_anchors(image, ground_truth, 7, 2)
    labels = torch.from_numpy(labels).unsqueeze(0)  # 设置该数据batch_size=1

    device = torch.device("cuda:0")
    model = YOLOv1Net(num_classes=2)
    # 在新版本的pytorch中，我们可以使用 model = model.to(device) 设置训练设备
    # 在某些旧版中，pytorch不可以通过上述方式设置，我们在Net中自行设置。
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)

    # loss = ??? 损失由4个值构成 中心位置损失 尺度损失 置信度损失  类别损失
    optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9)

    epochs = 10
    loop = tqdm(range(epochs), leave=False)
    for epoch in loop:
        optimizer.zero_grad()
        predict = model(features.float())
        loss = loss_bbox(predict, labels)
        loss.backward()
        optimizer.step()

        # print(loss.item())
        loop.set_description(f"epoch {epoch + 1}/{epochs}")
        loop.set_postfix(loss=loss.item())

