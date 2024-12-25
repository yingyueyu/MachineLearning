import numpy as np
import cv2 as cv

# ----------------平移-----------------------------------

img = cv.imread("../assets/example.png", 1)
rows, cols, channels = img.shape
M = np.float32([[1, 0, 100], [0, 1, 5]])
res = cv.warpAffine(img, M, (cols, rows))
cv.imshow('img', res)
cv.waitKey(0)
cv.destroyAllWindows()
