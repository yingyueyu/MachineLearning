import cv2
import numpy as np

img1 = cv2.imread("../assets/example.png")
tmp = np.zeros(img1.shape, dtype=np.uint8)
img2 = cv2.imread("../assets/hqyj.png")
h, w, c = img2.shape
tmp[:h, :w] = img2
b_channel = np.bitwise_or(img1[:, :, 0], tmp[:, :, 0])
g_channel = np.bitwise_or(img1[:, :, 1], tmp[:, :, 1])
r_channel = np.bitwise_or(img1[:, :, 2], tmp[:, :, 2])

result = cv2.merge((b_channel, g_channel, r_channel)).astype(np.uint8)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
