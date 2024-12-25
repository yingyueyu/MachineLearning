import numpy as np
import torch
import cv2

data = torch.load("./VOCdevkit/panels/ccpd_1k_224.pth", weights_only=False)
# cv2.imshow("features", data['features'][567].permute([1, 2, 0]).numpy())
test = data['labels'][567].numpy()
print(test[test < 1])
cv2.imshow("labels", data['labels'][567].numpy())

# a = np.zeros((100, 100),dtype=np.uint8)
# a[30:60, 30:60] = np.ones((30,30))
# cv2.imshow("test",a * 255)
cv2.waitKey(0)
