import cv2 as cv
import numpy as np

image = cv.imread("./imgs/logo.png")
h = np.shape(image)[0]
w = np.shape(image)[1]

grayimg = np.zeros((h,w,3),np.uint8)

for i in range(h):
    for j in range(w):
        grayimg[i,j] = (image[i,j][0] + image[i,j][1] + image[i,j][2]) / 3

cv.imshow("srcImage",image)
cv.imshow("grayImage",grayimg)

cv.waitKey(0)
cv.destroyAllWindows()
