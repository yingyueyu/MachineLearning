import cv2 as cv
import numpy as np
from time import time

image = cv.imread("../assets/example.png")
h, w, c = image.shape

start = time()

# 矩阵参与运算后速度会明显提升
grayimg = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
grayimg = grayimg.astype(np.uint8)

end = time()
print(end - start)

cv.imshow("srcImage", image)
cv.imshow("grayImage", grayimg)

cv.waitKey(0)
cv.destroyAllWindows()
