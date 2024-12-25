import cv2 as cv
import numpy as np

# ----------------图像简单放大-----------------------------------
img = cv.imread("../assets/example.png")
# h, w, c = img.shape
#
# img_zoom_out = np.zeros((h * 2, w * 2, c), dtype=np.uint8)
# img_zoom_out[0::2, 0::2] = img
# img_zoom_out[1::2, 1::2] = img
# cv.imshow('img', img_zoom_out)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
# 问题：
# 如果图像只缩小到200x200呢？
# ----------------插值法下的图像缩小-----------------------------------
h, w, c = img.shape
size = 500
new_img = np.zeros((size, size, 3), dtype=np.uint8)
sx = size / w
sy = size / h
for i in range(size):
    for j in range(size):
        new_i = round(i / sx)
        new_j = round(j / sx)
        new_img[j, i] = img[new_j, new_i]
cv.imshow("new_img", new_img)
cv.waitKey(0)
cv.destroyAllWindows()
