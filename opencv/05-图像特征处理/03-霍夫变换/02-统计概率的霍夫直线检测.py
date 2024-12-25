import cv2
import numpy as np

img = cv2.imread("./hough_lines.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
cv2.imshow("img", img)
cv2.waitKey(0)