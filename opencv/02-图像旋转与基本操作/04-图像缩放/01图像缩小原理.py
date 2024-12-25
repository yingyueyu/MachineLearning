import cv2 as cv
import numpy as np

# ----------------图像简单缩小-----------------------------------
img = cv.imread("../assets/example.png")

img_zoom_in = img[::2, ::2, :]
cv.imshow('img', img_zoom_in)

cv.waitKey(0)
cv.destroyAllWindows()
# 问题：
# 如果图像只缩小到200x200呢？
# ----------------插值法下的图像缩小-----------------------------------

