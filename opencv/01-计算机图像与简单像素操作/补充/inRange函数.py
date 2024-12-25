import numpy as np
import cv2

img = cv2.imread("./assets/inRange.png")

# 定义颜色范围（在HSV颜色空间中）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

# 将帧转换为HSV颜色空间
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 根据颜色范围创建掩膜
red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
# print(red_mask)
# cv2.imshow("red_mask", green_mask)
# cv2.waitKey(0)
# 对掩膜进行形态学操作，以去除噪声
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# print(red_mask.shape)
#
cv2.imshow("red_mask", blue_mask)
cv2.waitKey(0)
