import cv2
import numpy as np

img = cv2.imread("./hough_circle.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.Canny(gray, 50, 200)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 10)
print(circles)

cv2.imshow("img", gray)
cv2.waitKey(0)
