import cv2 as cv
import numpy as np
from time import time

image = cv.imread("../assets/example.png")
h, w, c = image.shape
grayimg = np.zeros((h, w), np.uint8)

start = time()

for i in range(h):
    for j in range(w):
        grayimg[i, j] = 0.114 * image[i, j][0] + 0.587 * image[i, j][1] + 0.299 * image[i, j][2]

end = time()
print(end - start)

cv.imshow("srcImage", image)
cv.imshow("grayImage", grayimg)

cv.waitKey(0)
cv.destroyAllWindows()
