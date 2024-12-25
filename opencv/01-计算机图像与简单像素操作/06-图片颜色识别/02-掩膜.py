import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../assets/colors.png")

# --- RGB -- 标准下的掩膜
# lowerb = np.array([1, 0, 0])
# upperb = np.array([255, 0, 0])
# mask = cv2.inRange(img, lowerb, upperb)

# -- HSV -- 标准下的掩膜
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lowerb = np.array([100, 43, 46])
upperb = np.array([124, 255, 255])
mask = cv2.inRange(img, lowerb, upperb)

kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])
mask = cv2.dilate(mask, kernel)
cv2.imshow("img", mask)
cv2.waitKey(0)
