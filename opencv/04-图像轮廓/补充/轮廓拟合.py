import cv2
import numpy as np

src = cv2.imread("./assets/signal.png")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 逼近多边形
adp = src.copy()
epsilon = 0.05 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result", adp)
cv2.waitKey(0)
cv2.destroyAllWindows()
