import numpy as np
import cv2

img = cv2.imread("assets/panel01.png")

# 定义颜色范围（在HSV颜色空间中）
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

# 对掩膜进行形态学操作，以去除噪声
kernel = np.ones((5, 5), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, 0,color=(0,0,255),thickness=2)
for item in contours[0]:
    x, y = item[0]
    cv2.circle(img, (x, y), 2, color=(0, 0, 255), thickness=2)

cv2.imshow("red_mask", img)
cv2.waitKey(0)
