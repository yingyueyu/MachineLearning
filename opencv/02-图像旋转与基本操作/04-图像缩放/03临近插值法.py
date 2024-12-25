# 面试中经常会问，如何将一个算子缩放
# 对于图像缩放，我们常用的插值方式
import cv2 as cv
import numpy as np

img = cv.imread("../assets/example.png")
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
