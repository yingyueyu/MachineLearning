import cv2 as cv
import numpy as np

img = cv.imread("../assets/example.png")
h, w, c = img.shape
size = 500
new_img = np.zeros((size, size, 3), dtype=np.uint8)
for i in range(size):
    for j in range(size):
        x = w / size * j
        y = h / size * i
        x1, y1 = (round(x - 1), round(y - 1))
        x2, y2 = (round(x + 1), round(y + 1))
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w - 1:
            x2 = w - 1
        if y2 > h - 1:
            y2 = h - 1
        if x1 != x2 or y1 != y2:
            q_11 = img[x1, y1]
            q_12 = img[x1, y2]
            q_21 = img[x2, y1]
            q_22 = img[x2, y2]
            r1 = (x2 - x) / (x2 - x1) * q_11 + (x - x1) / (x2 - x1) * q_21
            r2 = (x2 - x) / (x2 - x1) * q_12 + (x - x1) / (x2 - x1) * q_22
            p = (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2
        else:
            p = img(round(x1), round(y1))
        new_img[j, i] = p.astype(np.uint8)

cv.imshow("src", img)
cv.imshow("new_img", new_img)
cv.waitKey(0)
cv.destroyAllWindows()
