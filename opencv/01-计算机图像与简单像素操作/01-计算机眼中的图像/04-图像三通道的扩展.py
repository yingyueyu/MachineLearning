"""
绘制指定图像 大小为 280 x 280
"""
import cv2
import numpy as np

channel = np.zeros((280, 280, 3), dtype=np.uint8)
# 绘制指定区域
for i in range(7):
    channel[i * 40, :, :] = 255
    for j in range(7):
        channel[:, j * 40, :] = 255
        if (i + j == 6 or i == j) and i != 0 and i != 6:
            channel[i * 40:(i + 1) * 40, j * 40:(j + 1) * 40, 2] = 255
channel[-1, :, :] = 255
channel[:, -1, :] = 255

cv2.imshow("img", channel)
cv2.waitKey(0)
cv2.destroyAllWindows()
