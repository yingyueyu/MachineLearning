import torch
import cv2


data = torch.load("./data/seg/ccpd_3k.pth", weights_only=False)
cv2.imshow("features", data['features'][567].permute([1, 2, 0]).numpy())
cv2.imshow("labels", data['labels'][567].numpy())
cv2.waitKey(0)
