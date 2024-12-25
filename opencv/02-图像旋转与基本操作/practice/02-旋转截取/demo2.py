import cv2
import numpy as np
import math

origin_img = cv2.imread("./assets/rotate02.png")
h, w, c = origin_img.shape
a = min(w, h)
crop_origin_img = origin_img[:, w - a:]
M = cv2.getRotationMatrix2D((a / 2, a / 2), 105, 1.0)
dist_img = cv2.warpAffine(crop_origin_img, M, dsize=(a, a))

# 截取区域 200 x 200
crop_size = (200, 200)
x1 = int((a - crop_size[0]) / 2)
y1 = int((a - crop_size[1]) / 2)
x2 = x1 + crop_size[0]
y2 = y1 + crop_size[1]
dist_img = dist_img[y1:y2, x1:x2]

cv2.imshow("dist", dist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
