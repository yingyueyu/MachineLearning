import cv2
import numpy as np

src = cv2.imread("../assets/change_bg.png")
cv2.imshow("before", src)
img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(img, lower_blue, upper_blue)
_black = np.ones(blue_mask.shape, dtype=np.uint8) * 255

blue_channel = cv2.bitwise_or(src[:, :, 0], blue_mask)
green_channel = cv2.bitwise_or(src[:, :, 1], blue_mask)
red_channel = cv2.bitwise_or(src[:, :, 2], blue_mask)

src[:, :, 0] = blue_channel
src[:, :, 1] = green_channel
src[:, :, 2] = red_channel

cv2.imshow("after", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
