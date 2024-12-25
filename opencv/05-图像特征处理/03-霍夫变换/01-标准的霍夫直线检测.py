import cv2
import numpy as np

img = cv2.imread("./hough_lines.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLines(gray, 1, np.pi / 180, 150)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
cv2.imshow("img", img)
cv2.waitKey(0)
