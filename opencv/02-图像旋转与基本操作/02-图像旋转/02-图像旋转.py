import numpy as np
import cv2 as cv

# ------------------旋转-----------------------------------

img = cv.imread("../assets/example.png", 0)
rows, cols = img.shape[0:2]
# 第一个参数是旋转中心，第二个参数是旋转角度，第三个参数是旋转后的缩放因子
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 60, 1.2)
# 第三个参数是输出图像的尺寸中心，图像的宽和高
result = cv.warpAffine(img, M, (cols, rows))

cv.imshow('Result', result)

cv.waitKey(0)
cv.destroyAllWindows()
